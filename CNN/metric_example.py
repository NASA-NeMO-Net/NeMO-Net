from metrics import metrics
import numpy as np

a = np.random.randint(0, 10, size=[100, 32, 32, 1])
b = np.random.randint(0, 4, size=[100, 32, 32, 1])

metric = metrics(a, b, classes=10)
label_accuracy = metric.label_accuracy()
confusionMatrix = metric.print_confusionMatrix()
intersect_of_union = metric.intersect_of_union()
probability_of_detection = metric.probability_of_detection()
false_alarm_ratio = metric.false_alarm_ratio()
critical_success_index = metric.critical_success_index()
rmse = metric.rmse()
precision = metric.precision()
f1_score = metric.f1_score()

print(label_accuracy)
print(intersect_of_union)
print(probability_of_detection)
print(false_alarm_ratio)
print(critical_success_index)
print(rmse)
print(precision)
print(f1_score)
print(confusionMatrix)