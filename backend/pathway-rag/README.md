# Pathway-Compatible Docker RAG

This service provides the endpoints expected by RuraLens backend:

- `POST /v1/retrieve`
- `POST /v1/pw_ai_answer`
- `POST /v1/index/reload`
- `GET /health`

## Run

From repository root:

```bash
docker compose -f docker-compose.pathway.yml up -d --build
```

## Index Data

From `backend` folder, export docs from MongoDB:

```bash
npm run export-pathway
```

Then reload index in running container:

```bash
curl -X POST http://localhost:8000/v1/index/reload
```

## Auth

If `PATHWAY_MCP_TOKEN` is set, send:

`Authorization: Bearer <token>`

## Data Source

The container reads text docs from mounted directory:

`backend/pathway-data`
