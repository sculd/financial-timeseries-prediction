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
JOB_NAME=job_timeseries_${DATE}
gcloud ml-engine jobs submit training ${JOB_NAME} --package-path trainer --module-name trainer.task --job-dir gs://sonar-ml/${JOB_NAME} --runtime-version 1.6 --region us-central1 --scale-tier standard-1 -- --data_dir gs://sonar-ml/timeseries --output_dir gs://sonar-ml/${JOB_NAME} --train_steps 30000


gcloud ml-engine local train --package-path trainer --module-name trainer.task --job-dir gs://sonar-ml/job_timeseries -- --data_dir gs://sonar-ml/timeseries --output_dir gs://sonar-ml/job_timeseries --train_steps 1000


python trainer/task.py --job-dir gs://sonar-ml/job_timeseries --data_dir gs://sonar-ml/timeseries --output_dir gs://sonar-ml/job_timeseries --train_steps 10000 

python trainer/task.py --job-dir ./job_timeseries --data_dir ./ --output_dir ./job_timeseries --train_steps 100


