"""
Complete pipeline COPE-aligned:
- Filters export.xml for year 2021
- Parses GPX routes from workout-routes/
- Matches routes to workouts using overlap >=50% of shorter interval or midpoint within 10 min
- Links workouts to nearby Records (observations) by timestamp
- Builds COPE RDF graph (https://purl.archive.org/cope#)
- Exports Turtle, JSON-LD, NeoDash JSON, CSVs, Folium route maps, PyVis interactive KG
- Computes HRV (RMSSD) where HR timeseries available, step summaries, HR zones, anomalies
- Provides SPARQL query helper

Usage:
    python cope_full_pipeline_2021.py export.xml workout-routes/

Author:
    Asara Senaratne
"""
import sys, os, glob, json
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD, RDFS
import folium
from folium import PolyLine
import gpxpy
import matplotlib.pyplot as plt
from intervaltree import IntervalTree
import pandas as pd
from datetime import timedelta
from rdflib import URIRef, RDF
from rdflib import URIRef
import networkx as nx
from pyvis.network import Network

# ---------- Config constants----------
TARGET_YEAR = 2021 #this year was picked to keep results simple
COPE_BASE = "https://purl.archive.org/cope#"
COPE = Namespace(COPE_BASE)
EX = Namespace("https://purl.archive.org/cope/")
OUT_TTL = "cope_graph_2021.ttl"
OUT_JSONLD = "cope_graph_2021.jsonld"
OUT_PYVIS = "cope_graph_2021.html"
OUT_NEODASH = "cope_neodash_2021.json"
OUT_CSV = "cope_records_2021.csv"
OUT_ANOM = "cope_records_with_anomalies_2021.csv"
MAPS_DIR = "maps"
PLOTS_DIR = "plots"

OVERLAP_FRACTION_MIN = 0.5
MIDPOINT_TOLERANCE_SEC = 600
NEARBY_RECORD_SEC = 300

# ---------- Utilities ----------
def parse_iso(dt_str):
    if dt_str is None:
        return None
    s = dt_str.strip()
    # Normalize 'Z' to +00:00 if needed
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Fix stray space before tz like '...  +1100'
    if len(s) > 6 and s[-6] == " " and s[-5] in ("+", "-"):
        s = s[:-6] + s[-5:]
    try:
        return datetime.fromisoformat(s)
    except Exception:
        fmts = ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z")
        for f in fmts:
            try:
                return datetime.strptime(dt_str, f)
            except Exception:
                pass
    return None

def in_target_year(dt):
    return dt is not None and dt.year == TARGET_YEAR

def interval_seconds(a,b):
    if a is None or b is None: return None
    return (b - a).total_seconds()

def overlap_seconds(a1,a2,b1,b2):
    if None in (a1,a2,b1,b2):
        return 0
    start = max(a1,b1)
    end = min(a2,b2)
    return max(0, (end - start).total_seconds())

# ---------- Parse export.xml ----------
def parse_export_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records = []
    workouts = []
    for rec in root.findall("Record"):
        start = parse_iso(rec.get("startDate"))
        end = parse_iso(rec.get("endDate"))
        # keep if either start or end in target year
        if not (in_target_year(start) or in_target_year(end)):
            continue
        records.append({
            "type": rec.get("type"),
            "unit": rec.get("unit"),
            "value": rec.get("value"),
            "start": start,
            "end": end,
            "sourceName": rec.get("sourceName"),
            "sourceVersion": rec.get("sourceVersion"),
            "device": rec.get("device")
        })
    for wk in root.findall("Workout"):
        start = parse_iso(wk.get("startDate"))
        end = parse_iso(wk.get("endDate"))
        if not (in_target_year(start) or in_target_year(end)):
            continue
        workouts.append({
            "activityType": wk.get("workoutActivityType"),
            "duration": float(wk.get("duration")) if wk.get("duration") else None,
            "totalDistance": float(wk.get("totalDistance")) if wk.get("totalDistance") else None,
            "totalEnergyBurned": float(wk.get("totalEnergyBurned")) if wk.get("totalEnergyBurned") else None,
            "start": start,
            "end": end,
            "sourceName": wk.get("sourceName"),
            "sourceVersion": wk.get("sourceVersion"),
            "device": wk.get("device")
        })
    df_records = pd.DataFrame(records)
    return df_records, workouts

# ---------- Parse GPX routes ----------
def parse_gpx_folder(gpx_folder):
    routes = []
    for gpx_file in sorted(glob.glob(os.path.join(gpx_folder, "*.gpx"))):
        try:
            with open(gpx_file, "r", encoding="utf-8") as f:
                gpx = gpxpy.parse(f)
            points = []
            for trk in gpx.tracks:
                for seg in trk.segments:
                    for pt in seg.points:
                        points.append({"lat": pt.latitude, "lon": pt.longitude, "time": pt.time})
            times = [p["time"] for p in points if p["time"]]
            start = min(times) if times else None
            end = max(times) if times else None
            routes.append({"file": os.path.basename(gpx_file), "points": points, "start": start, "end": end})
        except Exception as e:
            print("GPX parse error:", gpx_file, e)
    return routes

# ---------- Matching route <-> workout ----------
def match_routes_workouts(routes, workouts, min_frac=OVERLAP_FRACTION_MIN, tol_sec=MIDPOINT_TOLERANCE_SEC):
    """
    Optimized matching of routes to workouts using sorted start times and interval filtering.
    Avoids nested O(n*m) loops where possible.
    """
    # --- Filter routes to year 2021 only ---
    routes_2021 = [
        (ri, r["start"], r["end"])
        for ri, r in enumerate(routes)
        if r["start"] and r["end"] and r["start"].year == 2021
    ]

    # If no 2021 routes, exit early
    if not routes_2021:
        print("No 2021 routes found.")
        return []

    # Precompute workout intervals
    workout_intervals = [
        (wi, w["start"], w["end"])
        for wi, w in enumerate(workouts)
        if w["start"] and w["end"]
    ]

    # Sort both lists by start time
    routes_2021.sort(key=lambda x: x[1])
    workout_intervals.sort(key=lambda x: x[1])

    matches = []
    wi_start_idx = 0

    for ri, rstart, rend in routes_2021:
        best = []

        for wi_idx in range(wi_start_idx, len(workout_intervals)):
            wi, wstart, wend = workout_intervals[wi_idx]

            # Workout starts after route ends → break (sorted)
            if wstart > rend + timedelta(seconds=tol_sec):
                break

            # Workout ends before route starts → skip ahead
            if wend + timedelta(seconds=tol_sec) < rstart:
                wi_start_idx = wi_idx + 1
                continue

            # Compute overlap
            ov = overlap_seconds(rstart, rend, wstart, wend)
            rlen = interval_seconds(rstart, rend)
            wlen = interval_seconds(wstart, wend)
            shorter = min(rlen, wlen) if rlen and wlen else None
            frac = (ov / shorter) if shorter and shorter > 0 else 0

            if shorter and frac >= min_frac:
                best.append({
                    "route_idx": ri,
                    "workout_idx": wi,
                    "method": "overlap",
                    "overlap_sec": ov,
                    "overlap_frac": frac
                })
                continue

            # Midpoint fallback
            rmid = rstart + (rend - rstart) / 2
            wmid = wstart + (wend - wstart) / 2
            delta = abs((rmid - wmid).total_seconds())

            if delta <= tol_sec:
                best.append({
                    "route_idx": ri,
                    "workout_idx": wi,
                    "method": "midpoint",
                    "delta_sec": delta
                })

        # Add the match or a no-match
        if best:
            matches.extend(best)
        else:
            matches.append({
                "route_idx": ri,
                "workout_idx": None,
                "method": "nomatch"
            })

    print(f"Processed {len(routes_2021)} routes from 2021.")
    return matches

# ---------- Link workouts to records by timestamp ----------
def link_workout_records(workouts, df_records, window_sec=NEARBY_RECORD_SEC):
    # Build interval tree of workouts expanded by ±window around start/end
    tree = IntervalTree()
    window = timedelta(seconds=window_sec)

    for wi, w in enumerate(workouts):
        wstart, wend = w["start"], w["end"]
        # Add three matching regions:
        #   1. whole workout interval
        #   2. window before start
        #   3. window after end
        tree.addi(wstart - window, wend + window, wi)

    links = []

    # Now scan records ONCE (450k iterations, but O(log 299) per query)
    for ri, r in df_records.iterrows():
        rstart = r["start"]
        if pd.isna(rstart):
            continue

        # Find all workouts whose interval intersects record start
        matches = tree.at(rstart)
        for m in matches:
            links.append({"workout_idx": m.data, "record_idx": ri})

    return links

# ---------- Build RDF graph (COPE aligned) ----------
def build_and_serialize_copegraph(df_records, workouts, routes, matches, links, out_ttl=OUT_TTL, out_jsonld=OUT_JSONLD):
    g = Graph()
    g.bind("cope", COPE)
    g.bind("ex", EX)
    g.bind("geo", Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#"))

    # Person
    person_uri = EX["person/me"]
    g.add((person_uri, RDF.type, COPE.Person))

    # Workouts as cope:Activity (PhysicalActivityEvent)
    for wi, w in enumerate(workouts):
        wid = f"workout/{wi}"
        subj = EX[wid]
        g.add((subj, RDF.type, COPE.Activity))
        if w.get("activityType"): g.add((subj, COPE.activityType, Literal(str(w["activityType"]))))
        if w.get("start"): g.add((subj, COPE.start, Literal(w["start"].isoformat(), datatype=XSD.dateTime)))
        if w.get("end"): g.add((subj, COPE.end, Literal(w["end"].isoformat(), datatype=XSD.dateTime)))
        if w.get("duration") is not None: g.add((subj, COPE.duration, Literal(str(w["duration"]))))
        # link person->workout
        g.add((person_uri, COPE.performed, subj))

    # Records as COPE Observation
    for ri, r in df_records.iterrows():
        rid = f"record/{ri}"
        subj = EX[rid]
        g.add((subj, RDF.type, COPE.Observation))
        if r.get("type"): g.add((subj, COPE.observedProperty, Literal(str(r["type"]))))
        if r.get("value") is not None: g.add((subj, COPE.value, Literal(str(r["value"]))))
        if r.get("unit"): g.add((subj, COPE.unit, Literal(str(r["unit"]))))
        if r.get("start"): g.add((subj, COPE.time, Literal(r["start"].isoformat(), datatype=XSD.dateTime)))
        # link person->observation
        g.add((person_uri, COPE.hasObservation, subj))

    # Routes as Trajectory and SpatialObservations for points
    for ri, r in enumerate(routes):
        rid = f"route/{ri}"
        rsubj = EX[rid]
        g.add((rsubj, RDF.type, COPE.Trajectory))
        if r.get("start"): g.add((rsubj, COPE.start, Literal(r["start"].isoformat(), datatype=XSD.dateTime)))
        if r.get("end"): g.add((rsubj, COPE.end, Literal(r["end"].isoformat(), datatype=XSD.dateTime)))
        # per-point spatial observations
        for pi, p in enumerate(r["points"]):
            p_uri = EX[f"{rid}/pt/{pi}"]
            g.add((p_uri, RDF.type, COPE.SpatialObservation))
            g.add((p_uri, Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#").lat, Literal(str(p["lat"]))))
            g.add((p_uri, Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#").long, Literal(str(p["lon"]))))
            if p.get("time"): g.add((p_uri, COPE.time, Literal(p["time"].isoformat(), datatype=XSD.dateTime)))
            # link trajectory -> point
            g.add((rsubj, COPE.hasPoint, p_uri))
        # link person->trajectory
        g.add((person_uri, COPE.hasTrajectory, rsubj))
    # Matches: route->workout associations
    for m in matches:
        ri = m["route_idx"]
        wi = m.get("workout_idx")
        ruri = EX[f"route/{ri}"]
        if wi is not None:
            wuri = EX[f"workout/{wi}"]
            g.add((ruri, COPE.associatedWith, wuri))
            # store method as literal
            if m.get("method"): g.add((ruri, COPE.matchMethod, Literal(m["method"])))
            if m.get("overlap_sec") is not None: g.add((ruri, COPE.overlapSeconds, Literal(str(m["overlap_sec"]))))
            if m.get("delta_sec") is not None: g.add((ruri, COPE.midpointDeltaSec, Literal(str(m["delta_sec"]))))
    # Links: workout -> observations (records)
    for l in links:
        wuri = EX[f"workout/{l['workout_idx']}"]
        ruri = EX[f"record/{l['record_idx']}"]
        g.add((wuri, COPE.hasObservation, ruri))
    # serialize
    g.serialize(destination=out_ttl, format="turtle")
    print("Saved RDF TTL:", out_ttl)
    # JSON-LD
    jld = g.serialize(format="json-ld", indent=2)
    with open(out_jsonld, "w", encoding="utf-8") as f:
        f.write(jld)
    print("Saved JSON-LD:", out_jsonld)
    return g

# ---------- NeoDash JSON export ----------
def rdf_to_networkx(rdf_graph):
    Gnx = nx.MultiDiGraph()
    for s, p, o in rdf_graph:
        s_str = str(s)
        Gnx.add_node(s_str)

        if isinstance(o, URIRef):
            o_str = str(o)
            Gnx.add_node(o_str)
            Gnx.add_edge(s_str, o_str, type=str(p))
    return Gnx
def export_neodash_json(G, out=OUT_NEODASH):
    nodes = []
    node_index = {}
    for i,(n,d) in enumerate(G.nodes(data=True)):
        node_index[n] = i
        nodes.append({"id": i, "label": d.get("label", str(n)), "properties": d})
    edges = []
    for u,v,k,d in G.edges(keys=True, data=True):
        # networkx multi-edges keys might be non-hashable; we assume simple
        edges.append({"from": node_index[u], "to": node_index[v], "label": d.get("type", ""), "properties": d})
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, default=str, indent=2)
    print("Saved NeoDash JSON:", out)
    return out

# ---------- PyVis interactive KG ----------
def export_pyvis_graph(rdf_graph, out_html=OUT_PYVIS, max_edges=20000):
    # ---------------------------
    # 1. Precompute rdf:type map
    # ---------------------------
    type_map = {}
    for s, p, o in rdf_graph.triples((None, RDF.type, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            type_map[str(s)] = str(o)

    # ---------------------------
    # 2. Build a filtered graph
    # ---------------------------
    Gnx = nx.MultiDiGraph()
    edge_count = 0

    for s, p, o in rdf_graph:
        if edge_count >= max_edges:
            break

        if not isinstance(s, URIRef):
            continue

        # HIGH-LEVEL FILTER:
        # Keep only ontology classes, workouts, and key record types.
        s_s = str(s)
        if ("Workout" not in s_s
            and "Record" not in s_s
            and "#Class" not in s_s
            and "#Property" not in s_s
            and "schema.org" not in s_s
            and "cope" not in s_s.lower()):
            continue

        Gnx.add_node(s_s)

        if isinstance(o, URIRef):
            o_s = str(o)
            if ("Workout" not in o_s
                and "Record" not in o_s
                and "#Class" not in o_s
                and "#Property" not in o_s
                and "schema.org" not in o_s
                and "cope" not in o_s.lower()):
                continue

            Gnx.add_node(o_s)
            Gnx.add_edge(s_s, o_s, label=str(p))
            edge_count += 1

    print("Filtered edges:", edge_count)

    # ---------------------------
    # 3. Assign labels & groups
    # ---------------------------
    for n in Gnx.nodes():
        label = n.split("/")[-1]
        t = type_map.get(n)
        group = t.split("#")[-1] if t else "Entity"
        Gnx.nodes[n]["label"] = label
        Gnx.nodes[n]["group"] = group

    # ---------------------------
    # 4. PyVis rendering
    # ---------------------------
    net = Network(height="800px", width="100%", directed=True)
    net.toggle_physics(True)

    for node, data in Gnx.nodes(data=True):
        net.add_node(node, label=data["label"], title=node, group=data["group"])

    for u, v, k, d in Gnx.edges(keys=True, data=True):
        net.add_edge(u, v, title=d.get("label", ""))

    net.write_html(out_html, open_browser=False)
    print("Saved PyVis KG:", out_html)
    return out_html

# ---------- Folium maps ----------
def save_route_maps(routes, matches, workouts, out_dir=MAPS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    # route maps
    for i, r in enumerate(routes):
        coords = [(p["lat"], p["lon"]) for p in r["points"] if p["lat"] is not None]
        if not coords: continue
        mean_lat = np.mean([c[0] for c in coords]); mean_lon = np.mean([c[1] for c in coords])
        fmap = folium.Map(location=[mean_lat, mean_lon], zoom_start=13)
        PolyLine(coords, color="blue", weight=4, opacity=0.7).add_to(fmap)
        folium.Marker(coords[0], popup=f"{r['file']} start").add_to(fmap)
        folium.Marker(coords[-1], popup=f"{r['file']} end").add_to(fmap)
        out_path = os.path.join(out_dir, f"route_{i}.html")
        fmap.save(out_path)
        print("Saved route map:", out_path)
    # matched route-workout maps
    for m in matches:
        if m.get("workout_idx") is None: continue
        ri = m["route_idx"]; wi = m["workout_idx"]
        r = routes[ri]; w = workouts[wi]
        coords = [(p["lat"], p["lon"]) for p in r["points"] if p["lat"] is not None]
        if not coords: continue
        mean_lat = np.mean([c[0] for c in coords]); mean_lon = np.mean([c[1] for c in coords])
        fmap = folium.Map(location=[mean_lat, mean_lon], zoom_start=13)
        PolyLine(coords, color="purple", weight=4, opacity=0.7).add_to(fmap)
        folium.Marker(coords[0], popup=f"{r['file']} start").add_to(fmap)
        folium.Marker(coords[-1], popup=f"{r['file']} end").add_to(fmap)
        folium.Marker([coords[0][0], coords[0][1]], popup=f"Workout start: {w['start']}").add_to(fmap)
        out_path = os.path.join(out_dir, f"route_{ri}_workout_{wi}.html")
        fmap.save(out_path)
        print("Saved matched map:", out_path)

# ---------- HRV & HR-zone computations ----------
def compute_hrv_and_zones(df_records, workouts, routes, out_plots=PLOTS_DIR):
    os.makedirs(out_plots, exist_ok=True)
    # extract heart rate records with timestamps
    hr_df = df_records[df_records["type"].str.contains("HeartRate", na=False)].copy()
    if hr_df.empty:
        print("No HeartRate records for HRV/zones.")
        return {}
    # ensure timestamp present
    if "start" not in hr_df.columns:
        print("No timestamps in HR records.")
        return {}
    # convert to numeric
    hr_df["value_num"] = pd.to_numeric(hr_df["value"], errors="coerce")
    # use observed max HR for zones
    max_hr = hr_df["value_num"].max(skipna=True)
    if pd.isna(max_hr):
        print("No numeric HR values.")
        return {}
    zones = {
        "zone1": (0, 0.5*max_hr),
        "zone2": (0.5*max_hr, 0.6*max_hr),
        "zone3": (0.6*max_hr, 0.7*max_hr),
        "zone4": (0.7*max_hr, 0.8*max_hr),
        "zone5": (0.8*max_hr, 1.0*max_hr)
    }
    results = {"max_hr": float(max_hr), "zones": zones}
    # For each workout, compute HRV RMSSD from HR samples whose timestamp in workout interval
    hrv_results = {}
    for wi, w in enumerate(workouts):
        wstart, wend = w["start"], w["end"]
        if wstart is None or wend is None: continue
        samples = hr_df[(hr_df["start"] >= wstart) & (hr_df["start"] <= wend)]
        vals = samples["value_num"].dropna().values
        times = pd.to_datetime(samples["start"]).values
        if len(vals) >= 2:
            # approximate RR intervals (seconds) = 60 / HR
            rr = 60.0 / vals
            diffs = np.diff(rr)
            rmssd = np.sqrt(np.mean(diffs**2))
            hrv_results[f"workout_{wi}"] = {"rmssd": float(rmssd), "n_samples": int(len(vals))}
            # plot HR time series
            plt.figure(figsize=(6,3))
            plt.plot(samples["start"], vals, marker='o')
            plt.title(f"Workout {wi} HR (n={len(vals)}), RMSSD={rmssd:.2f}s")
            plt.xlabel("Time"); plt.ylabel("HR (bpm)")
            png = os.path.join(out_plots, f"hr_workout_{wi}.png")
            plt.tight_layout(); plt.savefig(png); plt.close()
            print("Saved HR plot:", png)
    results["hrv_per_workout"] = hrv_results
    return results

# ---------- Anomaly detection (z-score per type) ----------
def detect_anomalies(df_records):
    df = df_records.copy()
    df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    df["anomaly"] = False
    for t, group in df.groupby("type"):
        vals = group["value_num"].dropna()
        if len(vals) < 5:
            continue
        mu, sigma = vals.mean(), vals.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            continue
        z = (group["value_num"] - mu) / sigma
        df.loc[group.index, "anomaly"] = z.abs() > 3.0
    return df

# ---------- SPARQL helper ----------
def run_sparql(rdflib_graph, sparql_query):
    res = rdflib_graph.query(sparql_query)
    cols = res.vars
    rows = []
    for row in res:
        rows.append([str(x) if x is not None else None for x in row])
    return pd.DataFrame(rows, columns=[str(c) for c in cols])

# ---------- Entrypoint pipeline ----------
def pipeline(export_xml, gpx_folder):
    print("Parsing export.xml...")
    df_records, workouts = parse_export_xml(export_xml)
    print(f"Records: {len(df_records)}, Workouts: {len(workouts)} (filtered year {TARGET_YEAR})")
    print("Parsing GPX folder...")
    routes = parse_gpx_folder(gpx_folder)
    print(f"Routes parsed: {len(routes)}")
    print("Matching routes to workouts...")
    matches = match_routes_workouts(routes, workouts)
    print("Linking workouts to nearby records...")
    links = link_workout_records(workouts, df_records)
    print(f"Links found: {len(links)}")
    print("Building RDF graph and serialising...")
    rdf_g = build_and_serialize_copegraph(df_records, workouts, routes, matches, links, out_ttl=OUT_TTL, out_jsonld=OUT_JSONLD)
    print("Export NeoDash JSON...")
    # Use RDF as source of triples for nodes (simple)
    Gnx = rdf_to_networkx(rdf_g)
    export_neodash_json(Gnx, out=OUT_NEODASH)
    print("Export interactive PyVis KG...")
    export_pyvis_graph(rdf_g, out_html=OUT_PYVIS)
    print("Save route and matched maps...")
    save_route_maps(routes, matches, workouts, out_dir=MAPS_DIR)
    print("Compute HRV and zones (if HR data present)...")
    hrres = compute_hrv_and_zones(df_records, workouts, routes, out_plots=PLOTS_DIR)
    print("Detect anomalies and save CSVs...")
    df_records.to_csv(OUT_CSV, index=False)
    df_anom = detect_anomalies(df_records)
    df_anom.to_csv(OUT_ANOM, index=False)
    print("Pipeline finished. Outputs:")
    print(" - Turtle:", OUT_TTL)
    print(" - JSON-LD:", OUT_JSONLD)
    print(" - PyVis:", OUT_PYVIS)
    print(" - NeoDash JSON:", OUT_NEODASH)
    print(" - CSVs:", OUT_CSV, OUT_ANOM)
    print(" - Maps in:", MAPS_DIR)
    print(" - Plots in:", PLOTS_DIR)
    return rdf_g, df_records, workouts, routes, matches, links, hrres

# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cope_full_pipeline_2021.py export.xml workout-routes/")
        sys.exit(1)
    export_xml = sys.argv[1]; gpx_folder = sys.argv[2]
    if not os.path.exists(export_xml):
        print("export.xml not found:", export_xml); sys.exit(1)
    if not os.path.isdir(gpx_folder):
        print("gpx folder not found:", gpx_folder); sys.exit(1)
    pipeline(export_xml, gpx_folder)
