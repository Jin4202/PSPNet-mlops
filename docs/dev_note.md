# Cloud Run Deployment Setup

## Goal
Deploy the PSPNet serving image to Cloud Run automatically on push to `main`, using GitHub Actions, without storing any long-lived GCP keys in the repo.

## Project
- GCP project: `bamboo-climate-497404-a7` (name: "MLOps", same project DVC already uses)
- Region: `us-central1`
- GitHub repo: `Jin4202/PSPNet-mlops`

## What was provisioned

**APIs enabled**: `run.googleapis.com`, `artifactregistry.googleapis.com`, `iamcredentials.googleapis.com`, `sts.googleapis.com`, `cloudresourcemanager.googleapis.com`

**Artifact Registry**: Docker repo `pspnet-serving` in `us-central1`, holds the serving images built from the project `Dockerfile`.

**Service account**: `github-ci-deploy@bamboo-climate-497404-a7.iam.gserviceaccount.com`, with roles:
- `roles/artifactregistry.writer` — push images
- `roles/iam.serviceAccountUser` — act as the Cloud Run runtime SA
- `roles/run.admin` — deploy revisions and set IAM policy (needed for `--allow-unauthenticated`; `roles/run.developer` alone lacks `setIamPolicy`)

**Workload Identity Federation** (no JSON key ever leaves GCP):
- Pool: `github-pool`
- Provider: `github-provider`, OIDC issuer `https://token.actions.githubusercontent.com`
- Attribute condition restricts the provider to `assertion.repository_owner == 'Jin4202'`
- `iam.workloadIdentityUser` binding on the service account is scoped to the exact repo principal: `principalSet://iam.googleapis.com/projects/472009295842/locations/global/workloadIdentityPools/github-pool/attribute.repository/Jin4202/PSPNet-mlops`

**GitHub repo variables** (set via `gh variable set`, not secrets — none of these values are sensitive on their own):
- `GCP_PROJECT_ID` = `bamboo-climate-497404-a7`
- `GCP_REGION` = `us-central1`
- `GCP_AR_REPO` = `pspnet-serving`
- `GCP_WIF_PROVIDER` = `projects/472009295842/locations/global/workloadIdentityPools/github-pool/providers/github-provider`
- `GCP_DEPLOY_SA` = `github-ci-deploy@bamboo-climate-497404-a7.iam.gserviceaccount.com`

## CI workflow change
Added a `deploy` job to `.github/workflows/ci.yml`, gated on `docker-build` succeeding and only running on `push` to `main`:
1. `google-github-actions/auth@v2` exchanges the GitHub OIDC token for short-lived GCP credentials via the WIF provider above — no static key in CI.
2. Builds the serving image and pushes it to Artifact Registry, tagged with `${{ github.sha }}`.
3. `google-github-actions/deploy-cloudrun@v2` deploys the image to the `pspnet-serving` Cloud Run service with `--allow-unauthenticated` (chosen for easy portfolio demoing — anyone with the URL can hit `/predict` and `/health`).

## Decisions worth remembering
- Chose WIF over a downloaded service-account key — avoids a long-lived secret sitting in GitHub, and the existing `gcp-key.json` in the repo (used for DVC) was already flagged as a footgun.
- Public Cloud Run access was a deliberate trade-off for demo convenience, not a production default — revisit if this becomes more than a portfolio project.
- Had to upgrade the deploy SA from `roles/run.developer` to `roles/run.admin` after discovering `run.developer` doesn't include `run.services.setIamPolicy`, which `--allow-unauthenticated` needs.

## Not yet done
- The `deploy` job is committed locally (`4a0731d`) but not pushed to `main` yet — first real deploy hasn't run.
- The GitHub PAT embedded in `git remote -v`'s origin URL was exposed in a session transcript and should be rotated.
