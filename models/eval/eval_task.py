import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from eval import Eval
from env.thor_env import ThorEnv

import torch
import constants
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import matplotlib.pyplot as plt
import random

# classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']
classes = ['0'] + constants.ALL_DETECTOR

import random
def loop_detection(vis_feats, actions, window_size=10):

    # not enough vis feats for loop detection
    if len(vis_feats) < window_size*2:
        return False, None

    nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90'] #, 'LookDown_15', 'LookUp_15']
    random.shuffle(nav_actions)

    start_idx = len(vis_feats) - 1

    for end_idx in range(start_idx - window_size, window_size - 1, -1):
        if (vis_feats[start_idx] == vis_feats[end_idx]).all():
            if all((vis_feats[start_idx-i] == vis_feats[end_idx-i]).all() for i in range(window_size)):
                return True, nav_actions[1] if actions[end_idx] == nav_actions[0] else nav_actions[0]

    return False, None


import math
def get_orientation(d):
    if d == 'left':
        h, v = -math.pi/2, 0.0
    elif d == 'up':
        h, v = 0.0, -math.pi/12
    elif d == 'down':
        h, v = 0.0, math.pi/12
    elif d == 'right':
        h, v = math.pi/2, 0.0
    else:
        h, v = 0.0, 0.0

    orientation = torch.cat([
        torch.cos(torch.ones(1)*(h)),
        torch.sin(torch.ones(1)*(h)),
        torch.cos(torch.ones(1)*(v)),
        torch.sin(torch.ones(1)*(v)),
    ]).unsqueeze(-1).unsqueeze(-1).repeat(1,7,7).unsqueeze(0).unsqueeze(0)

    return orientation

def get_panoramic_views(env):
    horizon = np.round(env.last_event.metadata['agent']['cameraHorizon'])
    rotation = env.last_event.metadata['agent']['rotation']
    position = env.last_event.metadata['agent']['position']

    # Left
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": (rotation['y'] + 270.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_left = Image.fromarray(np.uint8(env.last_event.frame))

    # Right
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": (rotation['y'] + 90.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_right = Image.fromarray(np.uint8(env.last_event.frame))

    # Up
    env.step({
        "action": "TeleportFull",
        "horizon": np.round(horizon - constants.AGENT_HORIZON_ADJ),
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_up = Image.fromarray(np.uint8(env.last_event.frame))

    # Down
    env.step({
        "action": "TeleportFull",
        "horizon": np.round(horizon + constants.AGENT_HORIZON_ADJ),
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_down = Image.fromarray(np.uint8(env.last_event.frame))

    # Left
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })

    return curr_image_left, curr_image_right, curr_image_up, curr_image_down


def get_panoramic_actions(env):
    action_pairs = [
        ['RotateLeft_90', 'RotateRight_90'],
        ['RotateRight_90', 'RotateLeft_90'],
        ['LookUp_15', 'LookDown_15'],
        ['LookDown_15', 'LookUp_15'],
    ]
    imgs = []
    actions = []

    curr_image = Image.fromarray(np.uint8(env.last_event.frame))

    for a1, a2 in action_pairs:
        t_success, _, _, err, api_action = env.va_interact(a1, interact_mask=None, smooth_nav=False)
        actions.append(a1)
        imgs.append(Image.fromarray(np.uint8(env.last_event.frame)))
        #if len(err) == 0:
        if curr_image != imgs[-1]:
            t_success, _, _, err, api_action = env.va_interact(a2, interact_mask=None, smooth_nav=False)
            actions.append(a2)
        else:
            #print(err)
            print('Error while {}'.format(a1))
    return actions, imgs



class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, successes, failures, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()


    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results):
        # reset model
        model.reset()

        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # extract language features
        feat = model.featurize([(traj_data, False)], load_mask=False)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        print()
        print('[', goal_instr, ']')
        for n, instr in enumerate(traj_data['turk_annotations']['anns'][r_idx]['high_descs']):
            print('  -', n+1, instr)
        print()

        maskrcnn = maskrcnn_resnet50_fpn(pretrained=True)
        
        anchor_generator = AnchorGenerator(
        sizes=tuple([(4, 8, 16, 32, 64, 128, 256, 512) for _ in range(5)]),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))
        maskrcnn.rpn.anchor_generator = anchor_generator

        # 256 because that's the number of features that FPN returns
        maskrcnn.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        # get number of input features for the classifier
        in_features = maskrcnn.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        maskrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 106)

        # now get the number of input features for the mask classifier
        in_features_mask = maskrcnn.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        maskrcnn.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        106)
        
        maskrcnn.eval()
        maskrcnn.load_state_dict(torch.load('mrcnn_alfred_all_004.pth'))
        maskrcnn = maskrcnn.cuda()

        prev_vis_feat = None
        prev_action = None
        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
        man_actions = ['PickupObject', 'SliceObject', 'OpenObject', 'PutObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff']

        prev_class = 0
        prev_center = torch.zeros(2)

        vis_feats = []
        pred_actions = []
        loop_count = 0

        #prev_subgoal = traj_data['plan']['high_pddl'][env.get_subgoal_idx() + 1]['discrete_action']['action'] # should be "GotoLocation"

        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        tt = 0
        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                break

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
            feat['frames'] = vis_feat
            vis_feats.append(vis_feat)
            if model.panoramic:
                #curr_image_left, curr_image_right, curr_image_up, curr_image_down = get_panoramic_views(env)
                panoramic_actions, imgs = get_panoramic_actions(env)
                curr_image_left, curr_image_right, curr_image_up, curr_image_down = imgs
                feat['frames_left'] = resnet.featurize([curr_image_left], batch=1).unsqueeze(0)
                feat['frames_right'] = resnet.featurize([curr_image_right], batch=1).unsqueeze(0)
                feat['frames_up'] = resnet.featurize([curr_image_up], batch=1).unsqueeze(0)
                feat['frames_down'] = resnet.featurize([curr_image_down], batch=1).unsqueeze(0)
                #t += len(panoramic_actions)
                #if t >= args.max_steps:
                #    break

            # forward model
            m_out = model.step(feat)
            #current_subgoal = traj_data['plan']['high_pddl'][env.get_subgoal_idx() + 1]['discrete_action']['action']
            #if prev_subgoal == 'GotoLocation' and current_subgoal != prev_subgoal: # find a target object
            #    print('find a target object')
            #    for a in nav_actions + ['<<stop>>']:
            #        m_out['out_action_low'][0,0,model.vocab['action_low'].word2index(a)] = -9999 # kill navigation actions
            #print(current_subgoal)
            m_pred = model.extract_preds(m_out, [(traj_data, False)], feat, clean_special_tokens=False)
            m_pred = list(m_pred.values())[0]

            # action prediction
            action = m_pred['action_low']

            # Loop detection
            isLoop, rand_action = loop_detection(vis_feats, pred_actions, 10)
            if isLoop:
                action = rand_action
                loop_count += 1

            if prev_vis_feat != None:
                od_score = ((prev_vis_feat - vis_feat)**2).sum().sqrt()
                epsilon = 1
                if od_score < epsilon:
                    dist_action = m_out['out_action_low'][0][0].detach().cpu()
                    dist_action = F.softmax(dist_action)
                    action_mask = torch.ones(len(model.vocab['action_low']), dtype=torch.float)
                    action_mask[model.vocab['action_low'].word2index(prev_action)] = -1
                    action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

            if action == cls.STOP_TOKEN:
                print("\tpredicted STOP")
                break

            pred_actions.append(action)

            # mask prediction
            mask = None
            if model.has_interaction(action):
                class_dist = m_pred['action_low_mask'][0]
                pred_class = np.argmax(class_dist)

                # mask generation
                with torch.no_grad():
                    out = maskrcnn([to_tensor(curr_image).cuda()])[0]
                    for k in out:
                        out[k] = out[k].detach().cpu()

                if sum(out['labels'] == pred_class) == 0:
                    #mask = np.zeros((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
                    # resample action and mask should be None
                    mask = None
                    
                    action_dist = m_out['out_action_low'][0][0]
                    action_mask = torch.zeros_like(action_dist)
                    for i in range(len(model.vocab['action_low'])):
                        if model.vocab['action_low'].index2word(i) in nav_actions[1:3]:
                            action_mask[i] = 1
                    action = model.vocab['action_low'].index2word(((action_dist - action_dist.min()) * action_mask).argmax().item())
                    print('      Resampled action:', action)
                else:
                    masks = out['masks'][out['labels'] == pred_class].detach().cpu()
                    scores = out['scores'][out['labels'] == pred_class].detach().cpu()

                    # Instance selection based on the minimum distance between the prev. and cur. instance of a same class.
                    if prev_class != pred_class:
                        scores, indices = scores.sort(descending=True)
                        masks = masks[indices]
                        prev_class = pred_class
                        prev_center = masks[0].squeeze(dim=0).nonzero().double().mean(dim=0)
                    else:
                        cur_centers = torch.stack([m.nonzero().double().mean(dim=0) for m in masks.squeeze(dim=1)])
                        distances = ((cur_centers - prev_center)**2).sum(dim=1)
                        distances, indices = distances.sort()
                        masks = masks[indices]
                        prev_center = cur_centers[0]

                    mask = np.squeeze(masks[0].numpy(), axis=0)

            # print action
            if args.debug:
                print(action)

            if model.has_interaction(action):
                print(t, action, classes[pred_class])
            else:
                print(t, action)

            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)

            if not t_success:
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

            prev_vis_feat = vis_feat
            prev_action = action

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True

        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / (float(t) + 1e-4))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / (float(t) + 1e-4))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward),
                     'loop_count': loop_count,}
        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = cls.get_metrics(successes, failures)

        print("-------------")
        print("SR: %d/%d = %.5f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("PLW SR: %.5f" % (results['all']['path_length_weighted_success_rate']))
        print("GC: %d/%d = %.5f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW GC: %.5f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = cls.get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        lock.release()

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res

    def create_stats(self):
            '''
            storage for success, failure, and results info
            '''
            self.successes, self.failures = self.manager.list(), self.manager.list()
            self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)