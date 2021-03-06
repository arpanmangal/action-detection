import torch
from torch import nn
import torch.nn.functional as F

from transforms import *
import torchvision.models

import itertools
from ops.ssn_ops import Identity, StructuredTemporalPyramidPooling


class SSN(torch.nn.Module):
    def __init__(self, num_class, num_tasks,
                 starting_segment, course_segment, ending_segment, modality,
                 base_model='resnet101', new_length=None,
                 dropout=0.8,
                 crop_num=1, no_regression=False, test_mode=False,
                 stpp_cfg=(1, (1, 2), 1), bn_mode='frozen',
                 task_head=False, glcu_skip=False,
                 glcu=False, additive_glcu=False,
                 verbose=True):
        super(SSN, self).__init__()
        self.modality = modality
        self.num_tasks = num_tasks
        self.num_segments = starting_segment + course_segment + ending_segment
        self.starting_segment = starting_segment
        self.course_segment = course_segment
        self.ending_segment = ending_segment
        self.reshape = True
        self.dropout = dropout
        self.crop_num = crop_num
        self.with_regression = not no_regression
        self.test_mode = test_mode
        self.bn_mode=bn_mode
        self.task_head = task_head
        self.glcu_skip = glcu_skip
        self.glcu = glcu
        self.additive_glcu = additive_glcu
        self.verbose = verbose

        if self.task_head:
            assert num_tasks > 0
        assert not (self.glcu and self.glcu_skip)
        if self.glcu or self.glcu_skip:
            assert self.task_head # Task head is necessary for having glcu
        if self.additive_glcu:
            assert self.glcu

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        if self.verbose:
            print(("""
    Initializing SSN with base model: {}.
    SSN Configurations:
        input_modality:     {}
        starting_segments:  {}
        course_segments:    {}
        ending_segments:    {}
        num_segments:       {}
        new_length:         {}
        dropout_ratio:      {}
        loc. regression:    {}
        bn_mode:            {}
        task_head:          {}
        glcu_skip:          {}
        glcu:               {}
        additive_glcu:      {}
        
        stpp_configs:       {} 
            """.format(base_model, self.modality,
                       self.starting_segment, self.course_segment, self.ending_segment,
                       self.num_segments, self.new_length, self.dropout, 'ON' if self.with_regression else "OFF",
                       self.bn_mode, self.task_head, self.glcu_skip, self.glcu, self.additive_glcu,
                       stpp_cfg)))

        self._prepare_base_model(base_model)

        self.base_feature_dim = self._prepare_ssn(num_class, num_tasks, stpp_cfg)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.prepare_bn()

    def _prepare_ssn(self, num_class, num_tasks, stpp_cfg):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, Identity())
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))

        self.stpp = StructuredTemporalPyramidPooling(feature_dim, True, configs=stpp_cfg)
        self.activity_fc = nn.Linear(self.stpp.activity_feat_dim(), num_class + 1)
        self.completeness_fc = nn.Linear(self.stpp.completeness_feat_dim(), num_class)

        nn.init.normal(self.activity_fc.weight.data, 0, 0.001)
        nn.init.constant(self.activity_fc.bias.data, 0)
        nn.init.normal(self.completeness_fc.weight.data, 0, 0.001)
        nn.init.constant(self.completeness_fc.bias.data, 0)

        self.test_fc = None
        if self.with_regression:
            self.regressor_fc = nn.Linear(self.stpp.completeness_feat_dim(), 2 * num_class)
            nn.init.normal(self.regressor_fc.weight.data, 0, 0.001)
            nn.init.constant(self.regressor_fc.bias.data, 0)
        else:
            self.regressor_fc = None

        if self.task_head:
            # Prepare task head
            self.task_head = GLCU(num_class, num_tasks, half_unit=True)
            self.task_head.init_weights()

        if self.glcu:
            # Prepare GLCU unit
            self.glcu = GLCU(feature_dim, num_tasks, additive_glcu=self.additive_glcu)
            self.glcu.init_weights()

        if self.glcu_skip:
            # Prepare GLCU unit across the classifier
            self.glcu_asc = HalfGLCU(self.stpp.activity_feat_dim(), num_tasks, middle_layers=[256])
            self.glcu_dsc_act = HalfGLCU(num_tasks, num_class + 1, bias=False)
            self.glcu_dsc_comp = HalfGLCU(num_tasks, num_class, bias=False)
            self.glcu_dsc_reg = HalfGLCU(num_tasks, 2 * num_class, bias=False)

        return feature_dim

    def prepare_bn(self):
        if self.bn_mode == 'partial':
            if self.verbose: print("Freezing BatchNorm2D except the first one.")
            self.freeze_count = 2
        elif self.bn_mode == 'frozen':
            if self.verbose:  print("Freezing all BatchNorm2D layers")
            self.freeze_count = 1
        elif self.bn_mode == 'full':
            self.freeze_count = None
        else:
            raise ValueError("unknown bn mode")

    def _prepare_base_model(self, base_model):
        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        elif base_model == 'InceptionV3':
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 299
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import model_zoo
            self.base_model = getattr(model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(SSN, self).train(mode)
        count = 0
        if self.freeze_count is None:
            return

        for m in self.base_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                count += 1
                if count >= self.freeze_count:
                    m.eval()

                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        
    def prepare_test_fc(self):
        # Test-time STPP+Classifier
        assert not self.glcu_skip
        self.test_fc = nn.Linear(self.activity_fc.in_features,
                                 self.activity_fc.out_features
                                 + self.completeness_fc.out_features * self.stpp.feat_multiplier
                                 + (self.regressor_fc.out_features * self.stpp.feat_multiplier if self.with_regression else 0))
        reorg_comp_weight = self.completeness_fc.weight.data.view(
            self.completeness_fc.out_features, self.stpp.feat_multiplier, self.activity_fc.in_features).transpose(0, 1)\
            .contiguous().view(-1, self.activity_fc.in_features)
        reorg_comp_bias = self.completeness_fc.bias.data.view(1, -1).expand(
            self.stpp.feat_multiplier, self.completeness_fc.out_features).contiguous().view(-1) / self.stpp.feat_multiplier

        weight = torch.cat((self.activity_fc.weight.data, reorg_comp_weight))
        bias = torch.cat((self.activity_fc.bias.data, reorg_comp_bias))

        if self.with_regression:
            reorg_reg_weight = self.regressor_fc.weight.data.view(
                self.regressor_fc.out_features, self.stpp.feat_multiplier, self.activity_fc.in_features).transpose(0, 1) \
                .contiguous().view(-1, self.activity_fc.in_features)
            reorg_reg_bias = self.regressor_fc.bias.data.view(1, -1).expand(
                self.stpp.feat_multiplier, self.regressor_fc.out_features).contiguous().view(-1) / self.stpp.feat_multiplier
            weight = torch.cat((weight, reorg_reg_weight))
            bias = torch.cat((bias, reorg_reg_bias))

        self.test_fc.weight.data = weight
        self.test_fc.bias.data = bias

    def prepare_test_fc_skip_glcu(self):
        # Test-time STPP+Classifier for skip-glcu
        # test_fc will also take the task as input now
        assert self.glcu_skip
        self.test_fc = nn.Linear(self.activity_fc.in_features + self.num_tasks,
                                 self.activity_fc.out_features
                                 + self.completeness_fc.out_features * self.stpp.feat_multiplier
                                 + (self.regressor_fc.out_features * self.stpp.feat_multiplier if self.with_regression else 0))
        """
        reorganizing act weight:
        [W_A] -> [W_A | W_T]
        """
        reorg_act_weight = torch.cat((self.activity_fc.weight.data, self.glcu_dsc_act.fcs[0].weight.data), dim=1)
        reorg_act_bias = self.activity_fc.bias.data

        """
        reorganizing comp weight:
        self.stpp.feat_multiplier == 3
        self.completeness_fc.weight.shape = [num_classes, 3 * 1024]
        after doing a view: [num_classes, 3, 1024]
        after transpose: [3, num_classes, 1024]
        after view: [3 * num_classes, 1024]
        i.e. transformed from [W1 | W2 | W3] -> [
                                                  W1
                                                  W2
                                                  W3
                                                ]
        With glcu_skip, 
        transforming from [W1 | W2 | W3] -> [
                                                  W1 | W_T / 3
                                                  W2 | W_T / 3
                                                  W3 | W_T / 3
                                                ]
        """
        reorg_comp_weight = self.completeness_fc.weight.data.view(
            self.completeness_fc.out_features, self.stpp.feat_multiplier, self.activity_fc.in_features).transpose(0, 1)\
            .contiguous().view(-1, self.activity_fc.in_features)
        stacked_comp_WT_by3 = self.glcu_dsc_comp.fcs[0].weight.data.repeat(self.stpp.feat_multiplier, 1).view(-1, self.num_tasks) / float(self.stpp.feat_multiplier)
        reorg_comp_weight = torch.cat((reorg_comp_weight, stacked_comp_WT_by3), dim=1)
        reorg_comp_bias = self.completeness_fc.bias.data.view(1, -1).expand(
            self.stpp.feat_multiplier, self.completeness_fc.out_features).contiguous().view(-1) / self.stpp.feat_multiplier

        weight = torch.cat((reorg_act_weight, reorg_comp_weight))
        bias = torch.cat((reorg_act_bias, reorg_comp_bias))

        if self.with_regression:
            """
            reorganizing reg weight:
            transforming from [W1 | W2 | W3] -> [
                                                  W1 | W_T / 3
                                                  W2 | W_T / 3
                                                  W3 | W_T / 3
                                                ]
            """
            reorg_reg_weight = self.regressor_fc.weight.data.view(
                self.regressor_fc.out_features, self.stpp.feat_multiplier, self.activity_fc.in_features).transpose(0, 1) \
                .contiguous().view(-1, self.activity_fc.in_features)
            stacked_reg_WT_by3 = self.glcu_dsc_reg.fcs[0].weight.data.repeat(self.stpp.feat_multiplier, 1).view(-1, self.num_tasks) / float(self.stpp.feat_multiplier)
            reorg_reg_weight = torch.cat((reorg_reg_weight, stacked_reg_WT_by3), dim=1)
            reorg_reg_bias = self.regressor_fc.bias.data.view(1, -1).expand(
                self.stpp.feat_multiplier, self.regressor_fc.out_features).contiguous().view(-1) / self.stpp.feat_multiplier
            weight = torch.cat((weight, reorg_reg_weight))
            bias = torch.cat((bias, reorg_reg_bias))

        self.test_fc.weight.data = weight
        self.test_fc.bias.data = bias


    def get_optim_policies(self, tune_glcu, tune_cls_head, tune_mid_glcu, tune_skip_glcu):
        """
        Tune GLCU is true for fine-tuning only the GLCU weights
        Tune CLS_HEAD is true for fine-tuning only cls-head weights
        """
        assert len([b for b in [tune_glcu, tune_cls_head, tune_mid_glcu, tune_skip_glcu] if b]) <= 1
        # assert not (tune_glcu and tune_cls_head)
        # assert not (tune_cls_head and tune_mid_glcu)
        # assert not (tune_glcu and tune_mid_glcu)

        if tune_glcu:
            for param in self.parameters():
                # Disable gradient computation for all layers
                param.requires_grad = False

            parameters = []
            for m in self.modules():
                if isinstance(m, GLCU):
                    ps = list(m.parameters())
                    parameters.extend(ps)
            for param in parameters:
                param.requires_grad = True
            return [
                {'params': parameters, 'lr_mult': 1, 'decay_mult': 1,
                'name': "glcu_parameters"}
            ]

        if tune_skip_glcu:
            for param in self.base_model.parameters():
                # Disable gradient computation for all backbone layers
                param.requires_grad = False

            glcu_weight = []
            glcu_bias = []
            for m in itertools.chain(self.glcu_asc.modules(), self.glcu_dsc_act.modules(), self.glcu_dsc_comp.modules(), self.glcu_dsc_reg.modules()):
                if isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    assert len(ps) <= 2
                    glcu_weight.append(ps[0])
                    if len(ps) == 2:
                        glcu_bias.append(ps[1])
                elif len(m._modules) == 0:
                    if len(list(m.parameters())) > 0:
                        print (m)
                        raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
            
            return [
                {'params': glcu_weight, 'lr_mult': 1, 'decay_mult': 1,
                'name': "glcu_weight"},
                {'params': glcu_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "glcu_bias"}
            ]

        if tune_cls_head:
            for param in self.parameters():
                # Disable gradient computation for all layers
                param.requires_grad = False

            act_cnt = 0
            comp_cnt = 0
            reg_cnt = 0
            glcu_cnt = 0
            for param in self.activity_fc.parameters():
                param.requires_grad = True
                act_cnt += 1
            for param in self.completeness_fc.parameters():
                param.requires_grad = True
                comp_cnt += 1
            if self.with_regression:
                for param in self.regressor_fc.parameters():
                    param.requires_grad = True
                    reg_cnt += 1
            if self.with_glcu:
                for fc in self.glcu.dfc:
                    for param in fc.parameters():
                        param.requires_grad = True
                        glcu_cnt += 1
            print ('#Act. parameters:', act_cnt)
            print ('#Comp. parameters:', comp_cnt)
            print ('#Reg. parameters:', reg_cnt)
            print ('#GLCU. parameters:', glcu_cnt)

            linear_weight = []
            linear_bias = []
            glcu_weight = []
            glcu_bias = []
            for m in itertools.chain(self.activity_fc.modules(), self.completeness_fc.modules(), self.regressor_fc.modules()):
                if isinstance(m, torch.nn.Linear):
                    ps = list(m.parameters())
                    assert len(ps) <= 2
                    linear_weight.append(ps[0])
                    if len(ps) == 2:
                        linear_bias.append(ps[1])
                else:
                    print (m)
                    raise ValueError("New module type")

            if self.with_glcu:
                for fc in self.glcu.dfc:
                    for m in fc.modules():
                        if isinstance(m, torch.nn.Linear):
                            ps = list(m.parameters())
                            assert len(ps) <= 2
                            glcu_weight.append(ps[0])
                            if len(ps) == 2:
                                glcu_bias.append(ps[1])
                        else:
                            print (m)
                            raise ValueError("New module type")
            
            return [
                {'params': linear_weight, 'lr_mult': 1, 'decay_mult': 1,
                'name': "linear_weight"},
                {'params': linear_bias, 'lr_mult': 2, 'decay_mult': 0,
                'name': "linear_bias"},
                {'params': glcu_weight, 'lr_mult': 10, 'decay_mult': 1,
                'name': "glcu_weight"},
                {'params': glcu_bias, 'lr_mult': 20, 'decay_mult': 0,
                'name': "glcu_bias"}
            ]

        if tune_mid_glcu:
            for param in self.base_model.parameters():
                # Disable gradient computation for all backbone layers
                param.requires_grad = False

            glcu_weight = []
            glcu_bias = []
            for fc in self.glcu.dfc:
                for m in fc.modules():
                    if isinstance(m, torch.nn.Linear):
                        ps = list(m.parameters())
                        assert len(ps) <= 2
                        glcu_weight.append(ps[0])
                        if len(ps) == 2:
                            glcu_bias.append(ps[1])
                    else:
                        print (m)
                        raise ValueError("New module type")
            
            return [
                {'params': glcu_weight, 'lr_mult': 10, 'decay_mult': 1,
                'name': "glcu_weight"},
                {'params': glcu_bias, 'lr_mult': 20, 'decay_mult': 0,
                'name': "glcu_bias"}
            ]


        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        linear_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                assert len(ps) <= 2
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                assert len(ps) <= 2
                linear_cnt += 1
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                # BN layers are all frozen in SSN
                bn_cnt += 1
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input, aug_scaling, target, reg_target, prop_type, task_target_input=None):
        if not self.test_mode:
            return self.train_forward(input, aug_scaling, target, reg_target, prop_type, task_target_input)
        else:
            return self.test_forward(input)

    def train_forward(self, input, aug_scaling, target, reg_target, prop_type, task_target_input=None):
        """
        For InceptionV3
        input.shape: [num_videos, 216, 299, 299]
        aug_scaling.shape: [num_videos, 8, 2]
        target.shape: [num_videos, 8]
        reg_target.shape: [num_videos, 8, 2]
        prop_type.shape: [num_videos, 8]
        prop_type[i, :] = [0, 1, 1, 1, 1, 1, 1, 2] == [FG, IN, BG]
        8 is the num_proposals
        216 = 8 * 9 * 3

        For BNInception:
        input.shape: [num_videos, 216, 224, 224]
        """
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        num_videos = input.size(0)

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # Input to base model has shape [num_videos * 8 * 9, 3, 224, 224]
        # Output shape == [num_videos * 8 * 9, 1024]
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))

        # Add GLCU unit
        glcu_task_pred = None
        if self.glcu:
            num_per_video = base_out.size(0) // num_videos
            step_features = base_out.view(num_videos, num_per_video, -1).mean(dim=1)
            gate, glcu_task_pred = self.glcu(step_features, task_target_input=task_target_input)
            gate = gate.repeat(1, num_per_video).view(-1, base_out.size(1))
            if self.additive_glcu:
                base_out = base_out + gate
            else:
                base_out = base_out * gate

        # activity_ft.shape = [num_videos * 8, 1024]
        # completeness_ft.shape = [num_videos * 8, 3072]
        activity_ft, completeness_ft = self.stpp(base_out, aug_scaling, [self.starting_segment,
                                                                         self.starting_segment + self.course_segment,
                                                                         self.num_segments])

        # Add GLCU unit
        if self.glcu_skip:
            num_per_video = activity_ft.size(0) // num_videos
            # Ascending phase
            step_features = activity_ft.view(num_videos, num_per_video, -1).mean(dim=1)
            glcu_task_pred = self.glcu_asc(step_features)
            
        # raw_act_fc.shape = [num_videos * 8, 780]. 780 == K+1 == Num Classes
        # raw_comp_fc.shape = [num_videos * 8, 779]
        raw_act_fc = self.activity_fc(activity_ft)
        raw_comp_fc = self.completeness_fc(completeness_ft)
        if self.glcu_skip:
            # Descending phase of skip-GLCU
            task_input = task_target_input if task_target_input is not None else F.softmax(glcu_task_pred, dim=1)
            act_refinement = self.glcu_dsc_act(task_input).repeat(1, num_per_video).view(-1, raw_act_fc.size(1))
            comp_refinement = self.glcu_dsc_comp(task_input).repeat(1, num_per_video).view(-1, raw_comp_fc.size(1))
            raw_act_fc = raw_act_fc + act_refinement
            raw_comp_fc = raw_comp_fc + comp_refinement

        # Add Task Head
        if self.task_head:
            # task_indexer = ((type_data == 0)).nonzero().squeeze()
            combined_scores = F.softmax(raw_act_fc[:, 1:], dim=1) * torch.exp(raw_comp_fc)
            combined_scores = combined_scores.view(num_videos, raw_act_fc.size(0) // num_videos, -1)
            combined_scores = torch.mean(combined_scores, dim=1)
            task_pred = self.task_head(combined_scores)
        else:
            task_pred = None

        type_data = prop_type.view(-1).data
        act_indexer = ((type_data == 0) + (type_data == 2)).nonzero().squeeze()
        comp_indexer = ((type_data == 0) + (type_data == 1)).nonzero().squeeze()
        target = target.view(-1)

        if self.with_regression:
            reg_target = reg_target.view(-1, 2)
            reg_indexer = (type_data == 0).nonzero().squeeze()
            raw_regress_fc = self.regressor_fc(completeness_ft).view(-1, self.completeness_fc.out_features, 2)
            if self.glcu_skip:
                # Descending phase of skip-GLCU
                task_input = task_target_input if task_target_input is not None else F.softmax(glcu_task_pred, dim=1)
                reg_refinement = self.glcu_dsc_reg(task_input).repeat(1, num_per_video).view(-1, raw_regress_fc.size(1), raw_regress_fc.size(2))
                raw_regress_fc = raw_regress_fc + reg_refinement

            return raw_act_fc[act_indexer, :], target[act_indexer], \
                   raw_comp_fc[comp_indexer, :], target[comp_indexer], \
                   raw_regress_fc[reg_indexer, :, :], target[reg_indexer], reg_target[reg_indexer, :], \
                   glcu_task_pred, task_pred
        else:
            return raw_act_fc[act_indexer, :], target[act_indexer], \
                   raw_comp_fc[comp_indexer, :], target[comp_indexer], \
                   glcu_task_pred, task_pred

    def test_forward(self, input):
        """
        input is of shape [num_frames * num_crops, 3, 224, 224]
        num_frames is by default 4 and num_crops is 10
        """
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        # base_out.shape == [40, 1024]
        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        return base_out
        # return self.test_fc(base_out), base_out

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    @property
    def with_task_head(self):
        return hasattr(self, 'task_head') and self.task_head

    @property
    def with_glcu(self):
        return self.with_glcu_skip or (hasattr(self, 'glcu') and self.glcu)

    @property
    def with_glcu_skip(self):
        return hasattr(self, 'glcu_skip') and self.glcu_skip 

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])



class GLCU(nn.Module):
    """
    Global Local Consistency Unit
    """

    def __init__ (self, in_feature_dim, num_tasks, middle_layers=[256], half_unit=False, additive_glcu=False, init_std=0.001):
        super (GLCU, self).__init__()

        self.init_std = init_std
        self.half_unit = half_unit
        self.additive_glcu = additive_glcu
        self.in_features = in_feature_dim

        # Map input to task
        self.ascending_layers = [in_feature_dim] + middle_layers + [num_tasks]
        self.afc = nn.ModuleList([])
        for in_size, out_size in zip(self.ascending_layers[:-1], self.ascending_layers[1:]):
            self.afc.append(nn.Linear(in_size, out_size))

        if not self.half_unit:
            # Map task back to input
            self.decending_layers = [num_tasks] + middle_layers[::-1] + [in_feature_dim]
            self.dfc = nn.ModuleList([])
            for in_size, out_size in zip(self.decending_layers[:-1], self.decending_layers[1:]):
                self.dfc.append(nn.Linear(in_size, out_size))

    def init_weights(self):
        for fc in self.afc:
            nn.init.normal(fc.weight.data, 0, self.init_std)
            nn.init.constant(fc.bias.data, 0)
        if not self.half_unit:
            for fc in self.dfc:
                nn.init.normal(fc.weight.data, 0, self.init_std)
                nn.init.constant(fc.bias.data, 0)

    def forward(self, feat, task_target_input=None):
        len_afc = len(self.afc)
        for i in range(len_afc - 1):
            feat = F.relu(self.afc[i](feat))
        task_feat = self.afc[-1](feat)

        if self.half_unit:
            return task_feat

        feat = F.softmax(task_feat, dim=1)
        if task_target_input is not None:
            task_input, use_full_task_target, target_ratio = task_target_input
            assert feat.size() == task_input.size()
            if use_full_task_target:
                feat = task_input
            else:
                feat = (feat * 1.0 + task_input * target_ratio) / (1.0 + target_ratio)

        len_dfc = len(self.dfc)
        for i in range(len_dfc - 1):
            feat = F.relu(self.dfc[i](feat))
        feat = self.dfc[-1](feat)

        if self.additive_glcu:
            return F.relu(feat), task_feat
        else:
            return F.sigmoid(feat), task_feat

    #  remember nn.functional.relu

class TC(nn.Module):
    """
    COIN's TC head
    """
    def __init__(self, W):
        super (TC, self).__init__()
        self.W = W
        self.in_features = W.shape[0]
        self.out_features = W.shape[1]

    def forward(self, step_scores):
        W = torch.autograd.Variable(torch.FloatTensor(self.W).cuda())
        return torch.mm(step_scores, W)


class HalfGLCU(nn.Module):
    """
    Either phase of the GLCU
    Simple MLP
    """
    def __init__ (self, in_feature_dim, out_feature_dim, middle_layers=[], bias=True, init_std=0.0001):
        super (HalfGLCU, self).__init__()

        self.init_std = init_std
        self.in_features = in_feature_dim
        self.out_features = out_feature_dim

        # Map input to task
        self.layers = [in_feature_dim] + middle_layers + [out_feature_dim]
        self.fcs = nn.ModuleList([])
        self.biases = [True] * (len(self.layers) - 1)
        self.biases[-1] = bias
        for in_size, out_size, b in zip(self.layers[:-1], self.layers[1:], self.biases):
            self.fcs.append(nn.Linear(in_size, out_size, bias=b))

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal(fc.weight.data, 0, self.init_std)
            nn.init.constant(fc.bias.data, 0)

    def forward(self, input_feat):
        feat = input_feat
        len_afc = len(self.fcs)
        for i in range(len_afc - 1):
            feat = F.relu(self.fcs[i](feat))
        return self.fcs[-1](feat)
