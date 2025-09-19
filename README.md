Deployment Steps:

Step 1: Login in Google Cloud from CLI:
First login to Google Cloud from CLI using 'gcloud auth login'.

Step 2: Assign Service account user role to the service account:
'gcloud projects add-iam-policy-binding automl-468818 --member=serviceAccount:467632688758@cloudbuild.gserviceaccount.com --role=roles/iam.serviceAccountUser'

Step 3: Assign Cloud Run role to the service account:
'gcloud projects add-iam-policy-binding automl-468818 --member=serviceAccount:467632688758@cloudbuild.gserviceaccount.com --role=roles/run.admin'

Step 4: Grant the required role to the Compute Engine default service account for accessing storage:
'gcloud projects add-iam-policy-binding automl-468818 --member="serviceAccount:467632688758-compute@developer.gserviceaccount.com" --role="roles/storage.admin"'

Step 5: Start the building and deployment process:
'gcloud builds submit --config cloudbuild.yaml --substitutions COMMIT_SHA=v2 --project=automl-468818'
