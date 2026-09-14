# Data contract

Original classroom videos, annotations, model weights, and student-level exam
scores were not distributed with the article. Nothing under `data/demo/` is
paper data.

Place real visual datasets in ImageFolder form:

```text
data/expression/{train,val,test}/{positive,neutral,negative}/image.jpg
data/body_behavior/{train,val,test}/{nodding,writing,turn_head,phone,sleeping}/image.jpg
```

Use student-level or video-level splits. Do not randomly distribute adjacent
frames from one student/video across train and test.

For raw grades, create a CSV with `student_id,group,score` and pass it to
`stats/reproduce_from_raw.py`.

