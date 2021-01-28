import open3d as o3d
import params
import torch
import torch.utils.data
import numpy as np
import gym
from gym import spaces
from pathlib import Path
import pyrender
import argparse
from PIL import Image

import modules
from FreeViewSynthesis import ext


class EnvTruckDiscrete(gym.Env):
    """
    The discrete environment for reconstructing the truck in tat dataset
    """
    def __init__(self,
                net_name,
                net_path,
                pw_dir,
                verbose=False,
                vis=False
                ):
                
        super(EnvTruckDiscrete, self).__init__()

        #set up output option
        self.verbose = verbose
        self.vis = vis

        # pw directory
        pw_dir = Path(pw_dir)

        self.src_im_paths = sorted(pw_dir.glob(f"im_*.jpg"))
        sample_im = np.array(Image.open(self.src_im_paths[0]))
        self.height, self.width, self.n_channels = sample_im.shape
        self.num_total_points = self.height * self.width

        self.src_Ks = np.load(pw_dir / "Ks.npy")
        self.src_Rs = np.load(pw_dir / "Rs.npy")
        self.src_ts = np.load(pw_dir / "ts.npy")

        src_dm_paths = sorted(pw_dir.glob("dm_*.npy"))
        dms = []
        for dm_path in src_dm_paths:
            dms.append(np.load(dm_path))        
        self.src_dms = np.array(dms)

        self.src_counts = np.load(pw_dir / "counts.npy")

        # set up basics fo the envrionments
        self.action_space = spaces.Discrete(params.N_DISCRETE_ACTIONS)
        self.action_map = params.action_map

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, self.n_channels), dtype=np.uint8)

        # mesh
        mesh_path = pw_dir.parent / "delaunay_photometric.ply"
        self.mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color((0.7, 0.7, 0.7))

        self.verts = np.asarray(self.mesh.vertices).astype(np.float32)
        self.faces = np.asarray(self.mesh.triangles).astype(np.int32)
        self.colors = np.asarray(self.mesh.vertex_colors).astype(np.float32)
        self.normals = np.asarray(self.mesh.vertex_normals).astype(np.float32)

        # set up FVS nework
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_width = 16
        self.n_nbs = 5
        self.invalid_depth_to_inf = True
        self.bwd_depth_thresh = 0.1

        if net_name == "fixed_identity_unet4.64.3":
            net_f = lambda: modules.get_fixed_net(
                enc_net="identity", dec_net="unet4.64.3", n_views=4
            )
        elif net_name == "fixed_vgg16unet3_unet4.64.3":
            net_f = lambda: modules.get_fixed_net(
                enc_net="vgg16unet3", dec_net="unet4.64.3", n_views=4
            )
        elif net_name == "aggr_vgg16unet3_unet4.64.3_mean":
            net_f = lambda: modules.get_aggr_net(
                enc_net="vgg16unet3", merge_net="unet4.64.3", aggr_mode="mean"
            )
        elif net_name == "rnn_identity_gruunet4.64.3":
            net_f = lambda: modules.get_rnn_net(
                enc_net="identity", merge_net="gruunet4.64.3"
            )
        elif net_name == "rnn_vgg16unet3_gruunet4.64.3_single":
            net_f = lambda: modules.get_rnn_net(
                enc_net="vgg16unet3", merge_net="gruunet4.64.3", mode="single"
            )
        elif net_name == "rnn_vgg16unet3_unet4.64.3":
            net_f = lambda: modules.get_rnn_net(
                enc_net="vgg16unet3", merge_net="unet4.64.3"
            )
        elif net_name == "rnn_vgg16unet3_gruunet4.64.3_nomasks":
            net_f = lambda: modules.get_rnn_net(
                enc_net="vgg16unet3", merge_net="gruunet4.64.3", cat_masks=False
            )
        elif net_name == "rnn_vgg16unet3_gruunet4.64.3_noinfdepth":
            net_f = lambda: modules.get_rnn_net(
                enc_net="vgg16unet3", merge_net="gruunet4.64.3"
            )
        elif net_name == "rnn_vgg16unet3_gruunet4.64.3":
            net_f = lambda: modules.get_rnn_net(
                enc_net="vgg16unet3", merge_net="gruunet4.64.3"
            )
        else:
            raise Exception("invalid net")
        
        if self.verbose:
            print("try loading network")
        self.net = net_f()
        state_dict = torch.load(str(net_path), map_location=self.device)
        self.net.load_state_dict(state_dict)
        if self.verbose:
            print("network loaded")

        # parameters for RL    
        self.step_cost = -0.1


    def step(self, action = 0):
        pose_new = self.action_map[action]

        image_new, dm_new, K_new, R_new, t_new = self.get_data_from_new_pose(pose_new)

        # get step output
        observation = image_new
        reward, done = self.get_reward(dm_new, K_new, R_new, t_new)
        info = {}

        # post process
        self.hist_dms = np.concatenate((self.hist_dms, dm_new[np.newaxis, :]))
        self.hist_Ks = np.concatenate((self.hist_Ks, K_new[np.newaxis, :]))
        self.hist_Rs = np.concatenate((self.hist_Rs, R_new[np.newaxis, :]))
        self.hist_ts = np.concatenate((self.hist_ts, t_new[np.newaxis, :]))
        
        return observation, reward, done, info


    def reset(self):
        pose_new = self.get_random_pose(self.action_space.n)

        image_new, dm_new, K_new, R_new, t_new = self.get_data_from_new_pose(pose_new)

        self.hist_dms = dm_new[np.newaxis, :]
        self.hist_Ks = K_new[np.newaxis, :]
        self.hist_Rs = R_new[np.newaxis, :]
        self.hist_ts = t_new[np.newaxis, :]

        observation= image_new
        return observation


    # def render(self, mode='human'):
    #     ...
    #     return


    # def close (self):
    #     ...
    #     return 

    
    def get_pose_change_cost(self, pose_new, pose_prev):
        """
        Get the cost for pose change
        """
        K_new, R_new, t_new = pose_new
        K_prev, R_prev, t_prev = pose_prev

        return self.step_cost * np.sum(np.abs(t_new - t_prev))

    
    def get_random_pose(self, n_actions):
        """
        Generate initial start pose
        """
        return self.action_map[np.random.randint(n_actions)]


    def get_data_from_new_pose(self, pose):
        """
        Use Free View Synthesis to generate new data from given pose, including image and count
        """
        # prepare data for image inference using FVS
        K, R, t = pose
        depth = self.render_depth_map_mesh(K, R, t, self.height, self.width)

        tgt_dm, tgt_K, tgt_R, tgt_t = depth, K, R, t

        src_Ks = self.src_Ks
        src_Rs = self.src_Rs
        src_ts = self.src_ts       
        src_dms = self.src_dms

        if self.verbose:
            print("running count_nbs")
        count = ext.preprocess.count_nbs(
            tgt_dm,
            tgt_K,
            tgt_R,
            tgt_t,
            src_dms,
            src_Ks,
            src_Rs,
            src_ts,
            bwd_depth_thresh=0.1,
        )
        if self.verbose:
            print("count")
            print(count)

        tgt_dm = self.pad(tgt_dm)

        # make nbs using count
        nbs = np.argsort(count)[::-1]
        nbs = nbs[: self.n_nbs]

        if self.verbose:
            print("print nbs")
            print(nbs)
        
        nb_src_dms = np.array([self.src_dms[ii] for ii in nbs])
        nb_src_dms = self.pad(nb_src_dms)

        nb_src_Ks = np.array([self.src_Ks[ii] for ii in nbs])
        nb_src_Rs = np.array([self.src_Rs[ii] for ii in nbs])
        nb_src_ts = np.array([self.src_ts[ii] for ii in nbs])

        if self.verbose:
            print("running get_sampling_map")
        patch = np.array((0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32)
        sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
            tgt_dm,
            tgt_K,
            tgt_R,
            tgt_t,
            nb_src_dms,
            nb_src_Ks,
            nb_src_Rs,
            nb_src_ts,
            patch,
            self.bwd_depth_thresh,
            self.invalid_depth_to_inf,
        )
        if self.verbose:
            print("get_sampling_map finished")

        # make data ready for network to infer
        data = {}
        data["tgt_dm"] = tgt_dm
        data["sampling_maps"] = sampling_maps
        data["valid_depth_masks"] = valid_depth_masks
        data["valid_map_masks"] = valid_map_masks

        nbs = np.argsort(count)[::-1]
        nbs = nbs[: self.n_nbs]

        srcs = np.array([self.pad(load_img(self.src_im_paths[ii])) for ii in nbs])
        data["srcs"] = srcs

        tgt_height = min(tgt_dm.shape[0], patch[1]) - patch[0]
        tgt_width = min(tgt_dm.shape[1], patch[3]) - patch[2]
        data["tgt"] = np.zeros((3, tgt_height, tgt_width), dtype=np.float32)

        # convert numpy array to tensor
        data_tensor = {}
        for k, v in data.items():
            v_tensor = torch.from_numpy(v).unsqueeze(0)
            data_tensor[k] = v_tensor.to(self.device).requires_grad_(requires_grad=False)

        if self.verbose:
            print("data check")
            for k, v in data_tensor.items():
                print("{} | {} | {}".format(k, v.dtype, v.shape))
                print(v)

        # network inference
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.net = self.net.to(self.device)
            self.net.eval()
            out = self.net(**data_tensor)

        # post process the output image
        im = out['out']
        im = im.detach().to("cpu").numpy()
        im = (np.clip(im, -1, 1) + 1) / 2
        im = im.transpose(0, 2, 3, 1)

        out_im = (255 * im[0]).astype(np.uint8)
        if self.verbose:
            print('im array')
            print(out_im)
            print('image shape')
            print(out_im.shape)
        if self.vis:
            Image.fromarray(out_im).save('obs.jpg')

        return out_im, depth, K, R, t

    
    def get_reward(self, tgt_dm, tgt_K, tgt_R, tgt_t, bwd_depth_thresh=0.1):
        """
        Calculate rewards based on number of new points and the cost for pose change
        """

        # calculate number of new points catched by the new view
        hist_dms = self.hist_dms
        hist_Ks = self.hist_Ks
        hist_Rs = self.hist_Rs
        hist_ts = self.hist_ts

        count_per_pix = ext.preprocess.count_nbs_per_pix(
            tgt_dm, 
            tgt_K, 
            tgt_R, 
            tgt_t, 
            hist_dms, 
            hist_Ks, 
            hist_Rs, 
            hist_ts,
            bwd_depth_thresh
        )

        num_new_points = np.sum(count_per_pix==0)
        percent_new_points = num_new_points / self.num_total_points
        print("percent of new points: {}".format(percent_new_points))

        # if not many new points are detected then done
        done = False
        if percent_new_points < 0.05:
            done = True

        # get pose change cost
        pose_new = (tgt_K, tgt_R, tgt_t)
        pose_prev = (hist_Ks[-1], hist_Rs[-1], hist_ts[-1])
        pose_change_cost = self.get_pose_change_cost(pose_new, pose_prev)
        print("pose change cost: {}".format(pose_change_cost))

        # calculate total rewards
        reward = percent_new_points + pose_change_cost

        return reward, done


    def render_depth_map_mesh(
        self,
        K,
        R,
        t,
        height,
        width,
        znear=0.05,
        zfar=1500,
    ):

        scene = pyrender.Scene()
        mesh = pyrender.Mesh(
            primitives=[
                pyrender.Primitive(
                    positions=self.verts,
                    normals=self.normals,
                    color_0=self.colors,
                    indices=self.faces,
                    mode=pyrender.GLTF.TRIANGLES,
                )
            ],
            is_visible=True,
        )
        mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        scene.add_node(mesh_node)

        cam = pyrender.IntrinsicsCamera(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            znear=znear,
            zfar=zfar,
        )
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = (-R.T @ t.reshape(3, 1)).ravel()
        cv2gl = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        T = T @ cv2gl
        cam_node = pyrender.Node(camera=cam, matrix=T)
        scene.add_node(cam_node)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
        light_node = pyrender.Node(light=light, matrix=np.eye(4))
        scene.add_node(light_node, parent_node=cam_node)

        render = pyrender.OffscreenRenderer(self.width, self.height)
        color, depth = render.render(scene)

        # if self.vis:
        #     depth[depth <= 0] = np.NaN
        #     depth = co.plt.image_colorcode(depth)
        #     imwrite(dm_path.with_suffix(".jpg"), depth)

        return depth


    def pad(self, im):
        if self.pad_width is not None:
            h, w = im.shape[-2:]
            mh = h % self.pad_width
            ph = 0 if mh == 0 else self.pad_width - mh
            mw = w % self.pad_width
            pw = 0 if mw == 0 else self.pad_width - mw
            shape = [s for s in im.shape]
            shape[-2] += ph
            shape[-1] += pw
            im_p = np.zeros(shape, dtype=im.dtype)
            im_p[..., :h, :w] = im
            im = im_p
        return im


def load_pickle_data(pickle_data_path):
    import pickle
    data = pickle.load(open(pickle_data_path, "rb"))
    return data


def load_img(p, height=None, width=None):
    im = Image.open(p)
    im = np.array(im)
    if (
        height is not None
        and width is not None
        and (im.shape[0] != height or im.shape[1] != width)
    ):
        raise Exception("invalid size of image")
    im = (im.astype(np.float32) / 255) * 2 - 1
    im = im.transpose(2, 0, 1)
    return im


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--net-name", type=str, required=True)
    # parser.add_argument("--net-path", type=str, required=True)
    # parser.add_argument("-d", "--pw-dir", type=str, required=True)
    # parser.add_argument("--verbose", action="store_true")
    # parser.add_argument("--vis", action="store_true")
    # args = parser.parse_args()

    # pw_dir = args.pw_dir
    # net_name = args.net_name
    # net_path = args.net_path
    # verbose = args.verbose
    # vis = args.vis

    from config import *

    print("init env starts")
    env_truck = EnvTruckDiscrete(net_name, net_path, pw_dir, verbose, vis)
    print("init env ends")

    print("reset env starts")
    out = env_truck.reset()
    print("reset env ends")

    print("run steps starts")
    for i in range(10):
        print("-------------------------------------------")
        print("iter = {}".format(i))
        print("-------------------------------------------")
        action = np.random.randint(env_truck.action_space.n)
        print("action: {}".format(action))
        observation, reward, done, info = env_truck.step(action)

        if done:
            break
    print("run steps ends")
