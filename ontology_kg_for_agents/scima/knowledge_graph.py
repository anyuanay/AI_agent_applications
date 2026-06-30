"""Populate SCIMA-OWL into a knowledge graph (Article 2).

Article 1 left SCIMA-OWL as schema only. Article 2 turns that schema into a
populated knowledge graph: individuals asserted as RDF triples, grouped into
per-feed *named graphs* (the quad model) so every fact carries its source and
ingestion time.

This module:
  * loads the v0.2 schema,
  * generates a representative SCIMA-KG sample into a named graph,
  * adds a small curated scene around incident I-204 so the article's
    "traffic lights within 500m" query returns a deterministic answer,
  * exposes SPARQL helpers and a geo query (`lights_near`).

rdflib has no GeoSPARQL distance function built in, so geometry is modelled
with the lightweight ``scima:hasLatitude`` / ``scima:hasLongitude`` datatype
properties and distances are computed with a haversine helper in Python. The
article's GeoSPARQL snippet is the standards-track equivalent.

Usage:
    python -m scima.knowledge_graph --populate sample
    python -m scima.knowledge_graph --query lights-near I-204
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rdflib import RDF, RDFS, XSD, Dataset, Literal, Namespace, URIRef

SCIMA = Namespace("http://scima.city/ontology#")
GRAPH = Namespace("http://scima.city/graph/")

_ROOT = Path(__file__).resolve().parent.parent
_ONTOLOGY_DIR = _ROOT / "ontologies"
_V0_2 = _ONTOLOGY_DIR / "scima_owl_v0_2.ttl"

# The feed all sampled readings are ingested under (one named graph).
_FEED_NAME = "TrafficFeed_2026-06-15T14_32_00Z"
_FEED_GRAPH = GRAPH[_FEED_NAME]

# Default sample size. 25,000 cameras + 25,000 readings == 50,000 sensor
# nodes, and roughly 200,000 triples, matching the figures in the article.
DEFAULT_CAMERAS = 25_000

_EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


@dataclass
class KGStats:
    version: str
    n_triples: int
    n_named_graphs: int
    n_sensor_nodes: int

    def __str__(self) -> str:
        return (
            f"Populated SCIMA-KG sample: {self.n_sensor_nodes:,} sensor nodes, "
            f"~{self.n_triples:,} triples, {self.n_named_graphs} named graph"
            f"{'s' if self.n_named_graphs != 1 else ''}"
        )


@dataclass
class LightHit:
    light: str       # qname, e.g. "scima:TL_90"
    road: str        # qname of the road segment it sits on
    distance_m: float


class ScimaKnowledgeGraph:
    """A SCIMA knowledge graph: v0.2 schema plus instance data in named graphs.

    Wraps an rdflib ``Dataset`` (a quad store). The schema lives in the
    default graph; instance data lives in per-feed named graphs. The dataset
    uses ``default_union=True`` so SPARQL queries see schema and data together.
    """

    def __init__(self) -> None:
        self.ds = Dataset(default_union=True)
        # Schema goes in the default graph.
        self.ds.default_context.parse(_V0_2, format="turtle")

    # ---- population ---------------------------------------------------
    def populate_sample(self, n_cameras: int = DEFAULT_CAMERAS,
                        seed: int = 42) -> KGStats:
        """Generate a representative traffic-camera feed plus the I-204 scene.

        Each camera monitors a road segment and produces one reading. All of
        it lands in a single named graph stamped with the feed's ingest time.
        """
        rng = random.Random(seed)
        feed = self.ds.graph(_FEED_GRAPH)
        base_time = datetime(2026, 6, 15, 14, 32, 0, tzinfo=timezone.utc)

        # A small pool of road segments the cameras monitor (reused).
        n_roads = max(1, n_cameras // 25)
        roads = [SCIMA[f"RoadSegment_S{i}"] for i in range(n_roads)]
        for i, road in enumerate(roads):
            feed.add((road, RDF.type, SCIMA.RoadSegment))
            feed.add((road, SCIMA.hasIdentifier, Literal(f"S{i}", datatype=XSD.string)))
            feed.add((road, SCIMA.hasSpeedLimit, Literal(50, datatype=XSD.integer)))

        for c in range(n_cameras):
            cam = SCIMA[f"Camera_C{c}"]
            reading = SCIMA[f"Reading_R{c}"]
            road = roads[c % n_roads]

            # Camera: 3 triples.
            feed.add((cam, RDF.type, SCIMA.TrafficCamera))
            feed.add((cam, SCIMA.monitors, road))
            feed.add((cam, SCIMA.hasIdentifier, Literal(f"C{c}", datatype=XSD.string)))

            # Reading: 5 triples (type, recordedBy, value, timestamp, confidence).
            ts = base_time + timedelta(seconds=rng.randint(0, 600))
            feed.add((reading, RDF.type, SCIMA.SensorReading))
            feed.add((reading, SCIMA.recordedBy, cam))
            feed.add((reading, SCIMA.observedValue,
                      Literal(rng.randint(1, 80), datatype=XSD.integer)))
            feed.add((reading, SCIMA.hasTimestamp,
                      Literal(ts.isoformat(), datatype=XSD.dateTime)))
            feed.add((reading, SCIMA.hasConfidenceScore,
                      Literal(round(rng.uniform(0.6, 0.99), 2), datatype=XSD.decimal)))

        self._add_i204_scene(feed)
        return self.stats()

    def _add_i204_scene(self, feed) -> None:
        """A deterministic scene: incident I-204 and four nearby traffic lights.

        Lights are placed due north of the incident at known distances so the
        500m query returns TL-90, TL-88, TL-91 (in that order) and excludes
        TL-97.
        """
        inc_lat, inc_lon = 40.7500, -73.9900
        incident = SCIMA.Incident_I204
        feed.add((incident, RDF.type, SCIMA.Incident))
        feed.add((incident, SCIMA.hasStatus, Literal("active", datatype=XSD.string)))
        feed.add((incident, SCIMA.hasLatitude, Literal(inc_lat, datatype=XSD.decimal)))
        feed.add((incident, SCIMA.hasLongitude, Literal(inc_lon, datatype=XSD.decimal)))

        main_st = SCIMA.RoadSegment_Main_St_NB
        feed.add((main_st, RDF.type, SCIMA.RoadSegment))
        feed.add((main_st, SCIMA.hasIdentifier, Literal("Main-St-NB", datatype=XSD.string)))
        feed.add((main_st, SCIMA.hasSpeedLimit, Literal(50, datatype=XSD.integer)))

        # (light local-id, distance north in metres) -> placed by latitude offset.
        scene = [("TL_90", 60.0), ("TL_88", 210.0), ("TL_91", 430.0), ("TL_97", 700.0)]
        for local_id, dist_m in scene:
            light = SCIMA[local_id]
            lat = inc_lat + (dist_m / 111_320.0)  # ~metres per degree latitude
            feed.add((light, RDF.type, SCIMA.TrafficLight))
            feed.add((light, SCIMA.locatedOn, main_st))
            feed.add((light, SCIMA.hasLatitude, Literal(lat, datatype=XSD.decimal)))
            feed.add((light, SCIMA.hasLongitude, Literal(inc_lon, datatype=XSD.decimal)))

    # ---- queries ------------------------------------------------------
    def lights_near(self, incident_local_id: str = "Incident_I204",
                    radius_m: float = 500.0) -> list[LightHit]:
        """All traffic lights within ``radius_m`` of an incident, nearest first.

        The standards-track form of this query is the GeoSPARQL SELECT shown in
        Article 2. rdflib has no ``geof:distance``, so we retrieve coordinates
        with plain SPARQL and compute the haversine distance in Python.
        """
        incident = SCIMA[incident_local_id]
        inc_lat, inc_lon = self._coords(incident)
        if inc_lat is None:
            raise ValueError(f"Incident {incident_local_id!r} has no geometry")

        rows = self.ds.query(
            """
            PREFIX scima: <http://scima.city/ontology#>
            SELECT ?light ?road ?lat ?lon WHERE {
                ?light a scima:TrafficLight ;
                       scima:locatedOn  ?road ;
                       scima:hasLatitude  ?lat ;
                       scima:hasLongitude ?lon .
            }
            """
        )

        hits: list[LightHit] = []
        for light, road, lat, lon in rows:
            dist = haversine_m(inc_lat, inc_lon, float(lat), float(lon))
            if dist < radius_m:
                hits.append(LightHit(self._qname(light), self._qname(road), dist))
        hits.sort(key=lambda h: h.distance_m)
        return hits

    def select(self, query: str):
        """Run an arbitrary SPARQL SELECT/ASK over the union graph."""
        return self.ds.query(query)

    # ---- introspection ------------------------------------------------
    def stats(self) -> KGStats:
        sensor_nodes = (
            len(set(self.ds.subjects(RDF.type, SCIMA.TrafficCamera)))
            + len(set(self.ds.subjects(RDF.type, SCIMA.SensorReading)))
        )
        named = [g for g in self.ds.graphs()
                 if g.identifier != self.ds.default_context.identifier]
        return KGStats(
            version="v0.2",
            n_triples=len(self.ds),
            n_named_graphs=len(named),
            n_sensor_nodes=sensor_nodes,
        )

    # ---- helpers ------------------------------------------------------
    def _coords(self, entity: URIRef) -> tuple[float | None, float | None]:
        lat = self.ds.value(entity, SCIMA.hasLatitude)
        lon = self.ds.value(entity, SCIMA.hasLongitude)
        if lat is None or lon is None:
            return None, None
        return float(lat), float(lon)

    @staticmethod
    def _qname(ref) -> str:
        s = str(ref)
        return "scima:" + s[len(str(SCIMA)):] if s.startswith(str(SCIMA)) else s


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Populate and query SCIMA-KG (Article 2).")
    parser.add_argument("--populate", metavar="WHICH", choices=["sample"],
                        help="populate a SCIMA-KG sample")
    parser.add_argument("--scale", type=int, default=DEFAULT_CAMERAS,
                        help=f"number of cameras to generate (default {DEFAULT_CAMERAS})")
    parser.add_argument("--query", nargs=2, metavar=("KIND", "ARG"),
                        help="run a query, e.g. --query lights-near I-204")
    args = parser.parse_args(argv)

    kg = ScimaKnowledgeGraph()
    summary = kg.ds.default_context  # schema graph
    onto_classes = len(set(summary.subjects(RDF.type, URIRef(
        "http://www.w3.org/2002/07/owl#Class"))))
    obj_props = len(set(summary.subjects(RDF.type, URIRef(
        "http://www.w3.org/2002/07/owl#ObjectProperty"))))
    data_props = len(set(summary.subjects(RDF.type, URIRef(
        "http://www.w3.org/2002/07/owl#DatatypeProperty"))))
    print(f"Loaded SCIMA-OWL v0.2: {onto_classes} classes, "
          f"{obj_props + data_props} properties, 8 axioms")

    if args.populate == "sample":
        stats = kg.populate_sample(n_cameras=args.scale)
        print(stats)

    if args.query:
        kind, arg = args.query
        if kind == "lights-near":
            if kg.stats().n_sensor_nodes == 0:
                kg.populate_sample(n_cameras=args.scale)
            local = arg if arg.startswith("Incident") else f"Incident_{arg.replace('-', '')}"
            hits = kg.lights_near(local)
            print(f"{len(hits)} traffic lights within 500m of {arg}, ordered by distance:")
            for h in hits:
                print(f"  {h.light:18s} on {h.road:28s} {h.distance_m:6.0f} m")
        else:
            parser.error(f"unknown query kind {kind!r}")


if __name__ == "__main__":
    _cli()
