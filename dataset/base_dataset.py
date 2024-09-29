import logging
import os
import random
from torch.utils.data import Dataset
import torch
import glob
from torch.nn.utils.rnn import pad_sequence
import re
import numpy as np

logger = logging.getLogger(__name__)

def lookat(center, target, up):
    """
    From: LAR-Look-Around-and-Refer
    https://github.com/eslambakr/LAR-Look-Around-and-Refer
    https://github.com/isl-org/Open3D/issues/2338
    https://stackoverflow.com/questions/54897009/look-at-function-returns-a-view-matrix-with-wrong-forward-position-python-im
    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    https://www.youtube.com/watch?v=G6skrOtJtbM
    f: forward
    s: right
    u: up
    """
    f = target - center
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = -s
    m[1, :-1] = u
    m[2, :-1] = f
    m[-1, -1] = 1.0

    t = np.matmul(-m[:3, :3], center)
    m[:3, 3] = t

    return m

def get_extrinsic(camera_location,target_location):
    camera_location=np.array(camera_location)
    target_location=np.array(target_location)
    up_vector = np.array([0, 0, -1])
    pose_matrix = lookat(camera_location, target_location, up_vector)
    pose_matrix_calibrated = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
    return pose_matrix_calibrated

SCANNET_ROOT = ""
SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt")  # scene_id, scene_id
def align_camera(pose_matrix, scene_id):
    """Adjust camera pose using the inverse of the mesh alignment matrix."""

    for line in open(SCANNET_META.format(scene_id, scene_id)).readlines():
        if 'axisAlignment' in line:
            axis_align_matrix = np.array(
                [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape((4, 4))
            break

    adjusted_pose = np.dot(axis_align_matrix, pose_matrix)
    return adjusted_pose

class BaseDataset(Dataset):

    def __init__(self):
        self.media_type = "point_cloud"
        self.anno = None
        self.attributes = None
        self.feats = None
        self.img_feats = None
        self.scene_feats = None
        self.scene_img_feats = None
        self.scene_masks = None
        self.feat_dim = 1024
        self.img_feat_dim = 1024

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def prepare_scene_features(self):
        scan_ids = set('_'.join(x.split('_')[:2]) for x in self.feats.keys())
        scene_feats = {}
        scene_img_feats = {}
        scene_masks = {}
        unwanted_words = ["wall", "ceiling", "floor", "object", "item"]
        for scan_id in scan_ids:
            if scan_id not in self.attributes:
                continue
            scene_attr = self.attributes[scan_id]
            obj_num = scene_attr['locs'].shape[0]
            obj_ids = scene_attr['obj_ids'] if 'obj_ids' in scene_attr else [_ for _ in range(obj_num)]
            obj_labels = scene_attr['objects']
            scene_feat = []
            scene_img_feat = []
            scene_mask = []
            for _i, _id in enumerate(obj_ids):
                item_id = '_'.join([scan_id, f'{_id:02}'])
                if item_id not in self.feats:
                    # scene_feat.append(torch.randn((self.feat_dim)))
                    scene_feat.append(torch.zeros(self.feat_dim))
                else:
                    scene_feat.append(self.feats[item_id])
                if self.img_feats is None or item_id not in self.img_feats:
                    # scene_img_feat.append(torch.randn((self.img_feat_dim)))
                    scene_img_feat.append(torch.zeros(self.img_feat_dim))
                else:
                    scene_img_feat.append(self.img_feats[item_id].float())
                if scene_feat[-1] is None or any(x in obj_labels[_id] for x in unwanted_words):
                    scene_mask.append(0)
                else:
                    scene_mask.append(1)
            scene_feat = torch.stack(scene_feat, dim=0)
            scene_img_feat = torch.stack(scene_img_feat, dim=0)
            scene_mask = torch.tensor(scene_mask, dtype=torch.int)
            scene_feats[scan_id] = scene_feat
            scene_img_feats[scan_id] = scene_img_feat
            scene_masks[scan_id] = scene_mask
        return scene_feats, scene_img_feats, scene_masks

    def get_anno(self, index):
        scene_id = self.anno[index]["scene_id"]
        if self.attributes is not None:
            scene_attr = self.attributes[scene_id]
            # obj_num = scene_attr["locs"].shape[0]
            scene_locs = scene_attr["locs"]
        else:
            scene_locs = torch.randn((1, 6))
        scene_feat = self.scene_feats[scene_id]
        if scene_feat.ndim == 1:
            scene_feat = scene_feat.unsqueeze(0)
        scene_img_feat = self.scene_img_feats[scene_id] if self.scene_img_feats is not None else torch.zeros((scene_feat.shape[0], self.img_feat_dim))
        scene_mask = self.scene_masks[scene_id] if self.scene_masks is not None else torch.ones(scene_feat.shape[0], dtype=torch.int)
        # assigned_ids = torch.randperm(200)[:len(scene_locs)]
        assigned_ids = torch.arange(len(scene_locs))

               
        #if camera pose exists, convert the scene_locs to camera coordinate
        if "camera" in self.anno[index]:

            camera_info = self.anno[index]["camera"]
            camera_position = camera_info["position"]
            lookat_point = camera_info["lookat"]
            # get extrinsic matrix
            view_matrix = get_extrinsic(camera_position, lookat_point)

            view_matrix=align_camera(view_matrix, scene_id)

            extrinsic_load = view_matrix
            camera_to_world = torch.from_numpy(extrinsic_load)
            world_to_camera = torch.inverse(camera_to_world)

            rotation_matrix = world_to_camera[:3, :3].permute(1, 0).unsqueeze(0)
            translation_vector = world_to_camera[:3, 3].reshape(-1, 1).permute(1, 0)

            temp=scene_locs[:, :3]

            temp=temp.to(torch.float32)
            rotation_matrix=rotation_matrix.to(torch.float32)
            translation_vector=translation_vector.to(torch.float32)

            X_c = temp@rotation_matrix#+translation_vector

            scene_locs[:, :3] = X_c

        return scene_id, scene_feat, scene_img_feat, scene_mask, scene_locs, assigned_ids
    

def update_caption(caption, new_ids):
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        old_id = int(caption[idx+4:idx+7])
        new_id = int(new_ids[old_id])
        caption = caption[:idx+4] + f"{new_id:03}" + caption[idx+7:]
    return caption


def recover_caption(caption, new_ids):
    old_ids = {new_id: i for i, new_id in enumerate(new_ids)}
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        new_id = int(caption[idx+4:idx+7])
        try:
            old_id = int(old_ids[new_id])
        except:
            old_id = random.randint(0, len(new_ids)-1)
        caption = caption[:idx+4] + f"{old_id:03}" + caption[idx+7:]
    return caption


def process_batch_data(scene_feats, scene_img_feats, scene_masks, scene_locs):
    # max_obj_num = max([e.shape[0] for e in scene_feats])
    # max_obj_num = 110
    # batch_size = len(scene_feats)
    batch_scene_feat = pad_sequence(scene_feats, batch_first=True)
    batch_scene_img_feat = pad_sequence(scene_img_feats, batch_first=True)
    batch_scene_mask = pad_sequence(scene_masks, batch_first=True)
    batch_scene_locs = pad_sequence(scene_locs, batch_first=True)
    # lengths = torch.tensor([len(feat) for feat in scene_feats])
    # max_obj_num = lengths.max()
    # batch_scene_mask = (torch.arange(max_obj_num).unsqueeze(0) < lengths.unsqueeze(1)).long()
    # batch_scene_feat = torch.zeros(batch_size, max_obj_num, scene_feats[0].shape[-1])
    # batch_scene_locs = torch.zeros(batch_size, max_obj_num, scene_locs[0].shape[-1])
    # batch_scene_colors = torch.zeros(batch_size, max_obj_num, scene_colors[0].shape[-2], scene_colors[0].shape[-1])
    # batch_scene_mask = torch.zeros(batch_size, max_obj_num, dtype=torch.long)
    # for i in range(batch_size):
    #     batch_scene_feat[i][:scene_feats[i].shape[0]] = scene_feats[i]
    #     batch_scene_locs[i][:scene_locs[i].shape[0]] = scene_locs[i]
    #     batch_scene_colors[i][:scene_colors[i].shape[0]] = scene_colors[i]
    #     batch_scene_mask[i][:scene_feats[i].shape[0]] = 1
    return batch_scene_feat, batch_scene_img_feat, batch_scene_locs, batch_scene_mask


def extract_all_ids(s):
    id_list = []
    for tmp in s.split('OBJ')[1:]:
        j = 0
        while tmp[:j+1].isdigit() and j < len(tmp):
            j = j + 1
        if j > 0:
            id_list.append(j)
    return id_list


if __name__ == "__main__":
    caption = "<OBJ001> <OBJ002>"
    assigned_ids = [1, 2, 3]
    caption = update_caption(caption, assigned_ids)
    print(caption)
    caption = recover_caption(caption, assigned_ids)
    print(caption)