from collections import OrderedDict

def _update_teacher_model(teacher_model, student_model, keep_rate=0.96):

    student_model_dict = student_model.state_dict()

    new_teacher_dict = OrderedDict()
    for key, value in teacher_model.state_dict().items():
        if key in student_model_dict.keys():
            new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
            )
        else:
            raise Exception("{} is not found in student model".format(key))

    teacher_model.load_state_dict(new_teacher_dict)