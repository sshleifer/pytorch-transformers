from typing import List

from torch import nn


def init_student(student, teacher):
    teacher_state_dict = teacher.state_dict()
    info = student.load_state_dict(teacher_state_dict, strict=False)
    assert info.missing_keys == [], info.missing_keys
    return student, info


def copy_decoder_layers(teacher, student, l2copy=[0, 2, 4, 7, 9, 11]):
    copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, l2copy)


def copy_layers(teacher_layers: nn.ModuleList, student_layers: nn.ModuleList, layers_to_copy: List) -> None:
    layers_to_copy = nn.ModuleList([l for i, l in enumerate(teacher_layers) if i in layers_to_copy])
    assert len(student_layers) == len(layers_to_copy), f"{len(student_layers)} != {len(layers_to_copy)}"
    student_layers.load_state_dict(layers_to_copy.state_dict())


def init_student_layer_from_avg(teacher_layers):
    student_state_dict = teacher_layers[0].state_dict()
    for l in teacher_layers[1:]:
        for k,v in l.state_dict().items():
            student_state_dict[k] += v
    return {k: v/len(teacher_layers) for k,v in student_state_dict.items()}


def make_wacky_student(teacher, layer_map, **config_kw):
    diff_dict =teacher.config.to_diff_dict()
    diff_dict.update(config_kw)
    diff_dict['n_decoder_layers'] = len(layer_map)
    student_cfg = type(teacher.config)(**diff_dict)
    student = type(teacher)(student_cfg)
    for stu_layer, teacher_layers in layer_map.items():
        teacher_layers = [x for i,x in enumerate(teacher.model.decoder.layers) if i in teacher_layers]
        student_state_dict = init_student_layer_from_avg(teacher_layers)
        student.model.decoder.layers[stu_layer].load_state_dict(student_state_dict)
    return student


