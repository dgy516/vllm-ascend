# Review and Implement Jenkins DeployCase CI

Use `.ci/deploy_cases/*.yaml` as the source of truth for generated deployment docs,
Jenkins deploy smoke runs, benchmark placeholders, accuracy execute-only checks, and
nightly HTML/JUnit/CSV reports.

Keep runtime outputs in `reports/` and `logs/`. Do not store credentials in `.ci/`.
