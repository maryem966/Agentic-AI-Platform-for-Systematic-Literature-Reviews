"""
Templates Cypher paramétrés — exhaustifs et pré-approuvés.

RÈGLE ABSOLUE DE SÉCURITÉ (Section 10.4 du CdC) :
  - Le LLM ne génère JAMAIS directement une requête Cypher.
  - Seuls les templates de ce fichier sont exécutés sur Neo4j.
  - La connexion Neo4j est en LECTURE SEULE pour le retrieval.
  - Tout nouveau template doit être ajouté ici ET revu par un humain.

Usage :
  from src.config.cypher_templates import CYPHER_TEMPLATES, ALLOWED_TEMPLATE_NAMES

  if template_name not in ALLOWED_TEMPLATE_NAMES:
      raise ValueError(f"Template Cypher non autorisé : {template_name}")

  cypher = CYPHER_TEMPLATES[template_name]
  results = await neo4j_read_only.query(cypher, params)
"""

CYPHER_TEMPLATES: dict[str, str] = {
    # Trouver les publications qui étudient un concept
    "concept_to_publications": """
        MATCH (c:Concept {normalized_name: $concept})<-[:STUDIES_CONCEPT]-(p:Publication)
        RETURN p.id AS pub_id, p.title AS title
        LIMIT $limit
    """,
    # Trouver les datasets utilisés avec une méthode donnée
    "method_to_datasets": """
        MATCH (m:Method {name: $method})<-[:USES_METHOD]-(p:Publication)
              -[:USES_DATASET]->(d:Dataset)
        RETURN d.name AS dataset, p.id AS pub_id
        LIMIT $limit
    """,
    # Toutes les relations d'une publication
    "publication_relations": """
        MATCH (p:Publication {id: $pub_id})-[r]->(n)
        RETURN type(r) AS relation_type, labels(n) AS node_labels,
               n.name AS node_name, n.id AS node_id
        LIMIT $limit
    """,
    # Publications utilisant deux méthodes en même temps (comparaison)
    "method_comparison": """
        MATCH (m1:Method {name: $method1})<-[:USES_METHOD]-(p:Publication)
              -[:USES_METHOD]->(m2:Method {name: $method2})
        RETURN p.id AS pub_id, p.title AS title
        LIMIT $limit
    """,
    # Publications utilisant un dataset spécifique
    "dataset_publications": """
        MATCH (d:Dataset {name: $dataset})<-[:USES_DATASET]-(p:Publication)
        RETURN p.id AS pub_id, p.title AS title
        LIMIT $limit
    """,
    # Résultats et métriques d'une publication
    "publication_metrics": """
        MATCH (p:Publication {id: $pub_id})-[:REPORTS_RESULT]->(r:Result)
              -[:USES_METRIC]->(m:Metric)
        RETURN m.name AS metric, r.value AS value,
               r.evidence_id AS evidence_id
        LIMIT $limit
    """,
    # Publications qui contredisent une publication donnée
    "contradicting_publications": """
        MATCH (p1:Publication {id: $pub_id})-[:CONTRADICTS]->(p2:Publication)
        RETURN p2.id AS pub_id, p2.title AS title
        LIMIT $limit
    """,
    # Publications qui soutiennent une publication donnée
    "supporting_publications": """
        MATCH (p1:Publication {id: $pub_id})-[:SUPPORTS]->(p2:Publication)
        RETURN p2.id AS pub_id, p2.title AS title
        LIMIT $limit
    """,
    # Limitations signalées dans une publication
    "publication_limitations": """
        MATCH (p:Publication {id: $pub_id})-[:HAS_LIMITATION]->(l:Limitation)
        RETURN l.text AS limitation_text, l.evidence_id AS evidence_id
        LIMIT $limit
    """,
}

# Ensemble immuable des noms de templates autorisés
ALLOWED_TEMPLATE_NAMES: frozenset[str] = frozenset(CYPHER_TEMPLATES.keys())
