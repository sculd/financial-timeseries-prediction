gcloud ml-engine jobs submit training job_timeseries \
--package-path trainer \
--module-name trainer.task \
--job-dir gs://sonar-ml/job_timeseries \
--data_dir gs://sonar-ml/timeseries \
--runtime-version 1.2 \
--region us-central1 \
-- \
--output_dir gs://sonar-ml/job_timeseries \
--train_steps 10000


DATE=`date '+%Y_%m_%dT%H_%M_%S'`
gcloud ml-engine jobs submit training job_timeseries_${DATE} --package-path trainer --module-name trainer.task --job-dir gs://sonar-ml/job_timeseries --runtime-version 1.6 --region us-central1 -- --data_dir gs://sonar-ml/timeseries --output_dir gs://sonar-ml/job_timeseries --train_steps 10000 


gcloud ml-engine local train --package-path trainer --module-name trainer.task --job-dir gs://sonar-ml/job_timeseries -- --data_dir gs://sonar-ml/timeseries --output_dir gs://sonar-ml/job_timeseries --train_steps 10000 
