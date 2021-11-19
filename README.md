# AASC Sentence Search API

updated version of https://github.com/HelloRusk/AASC_sentence_search

https://user-images.githubusercontent.com/36184621/124714281-8823a000-df3c-11eb-97e5-ca41685208a1.mov

## Local Setup

1. Clone this repo
2. Download `AASC_embeddings.npz` and `AASC_embeddings.tsv` from https://drive.google.com/drive/folders/18qHqTuBUSQ1lYD0otjvh_OtuVeCQE17h, and put them into `/data`
3. Install dependencies via `poetry install`
4. Run `poetry run uvicorn main:app --reload`, wait for the log `Application startup complete.`, and open `http://localhost:8000`
5. Click `Try it out` button