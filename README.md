# IITMBS_MLOPS_OPPE2 


- With the help of explainability tools, describe in plain English the factors on which the samples that have heart disease not predicted are most dependent on. (1 mark)

- Test for fairness with “age” as the sensitive attribute using fairlearn. (bucket the age into bin size 20 years if needed)  (1 mark)

- Convert the provided notebook to a dockerized, API-deployed model execution on GCP. Use k8s with auto scaling (max pod-3) as a deployment layer (4 marks - CI and CD should be mandatorily triggered using GitHub Actions/Workflows)

- Demonstrate per sample prediction along with logging, observability. Use a 100-row randomly generated data for this task. (2 marks)

- Performance monitoring and request timeout analysis with a high concurrency(>2000) workload using wrk. You can use the same random sample as generated in the earlier step. (1 mark)

- Compute input drift comparing trained data with the generated data used for prediction in the earlier step. (1 mark)


# Fairness Comment (Age as Sensitive Attribute)

- Fairness analysis was performed using Fairlearn with age as the sensitive attribute, bucketed into 20-year age groups.

- The model shows consistent accuracy across all age groups (21–40, 41–60, 61–80), with accuracy values close to the overall performance.

- The false negative rate is 0.0 for all age groups, indicating that the model does not disproportionately fail to detect heart disease for any particular age group.

- Based on these metrics, the model does not exhibit age-based unfairness under the evaluated fairness criteria.