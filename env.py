import params

import torch
import torch.utils.data
import numpy as np
import gym
from gym import spaces
import open3d as o3d
from pathlib import Path
import pyrender
from FreeViewSynthesis import ext
from FreeViewSynthesis.exp import modules
from FreeViewSynthesis.exp.dataset import load



class EnvTruckDiscrete(gym.Env):
    """
    The environment for reconstructing the truck in tat dataset
    """
    def __init__(self,
                height,
                weight,
                n_channels,
                pw_dir,
                net_name
                ):
                
        super(EnvTruckDiscrete, self).__init__()

        # set up basics fo the envrionments
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.action_map = params.action_map

        self.height = height
        self.weight = weight
        self.n_channels = n_channels
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.weight, self.n_channels), dtype=np.uint8)

        self.start_pose = self.get_start_pose(self.action_space.n)
        self.pose_prev = self.start_pose
        self.num_total_points = counts.shape[0]

        # set up FVS nework
        self.eval_device = "cuda:0"
        self.pad_width = 16
        self.n_nbs = 5
        self.invalid_depth_to_inf = True
        self.bwd_depth_thresh=0.1

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

        self.net = net_f()

        # pw directory
        pw_dir = Path(pw_dir)

        self.src_im_paths = sorted(pw_dir.glob(f"im_*.jpg"))

        self.src_Ks = np.load(pw_dir / "Ks.npy")
        self.src_Rs = np.load(pw_dir / "Rs.npy")
        self.src_ts = np.load(pw_dir / "ts.npy")

        src_dm_paths = sorted(pw_dir.glob("dm_*.npy"))
        self.src_dms = load_depth_maps(src_dm_paths)

        # mesh
        mesh_path = pw_dir.parent / "delaunay_photometric.ply"
        self.mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color((0.7, 0.7, 0.7))

        self.verts = np.asarray(self.mesh.vertices).astype(np.float32)
        self.faces = np.asarray(self.mesh.triangles).astype(np.int32)
        self.colors = np.asarray(self.mesh.vertex_colors).astype(np.float32)
        self.normals = np.asarray(self.mesh.vertex_normals).astype(np.float32)

        # parameters for RL    
        self.step_cost = -0.1
        self.seen_point_list = []


    def step(self, action = 0):
        pose_new = self.action_map[action]
        pose_change = pose_new - self.pose_prev

        image_new, count = self.get_new_data_from_pose(pose_new)

        observation = image_new
        reward = self.get_reward(count, pose_change)
        done = self.check_done()

        self.pose_prev = pose_new
        
        return observation, reward, done, info


    def reset(self):
        poes_new = self.get_start_pose(self.n_discrete_actions)
        self.start_pose = pose_new
        self.pose_prev = self.start_pose

        image_new, count = get_new_data_from_pose(pose_new)
        observation= image_new
        return observation


    # def render(self, mode='human'):
    #     ...
    #     return


    # def close (self):
    #     ...
    #     return 

    
    def get_start_pose(self, n_actions):
        """
        Generate initial start pose
        """
        return self.action_map[np.random.randint(n_actions)]


    def get_new_data_from_pose(self, pose):
        """
        Use Free View Synthesis to generate new data from given pose, including image and count
        """
        # prepare data for image inference using FVS
        K, R, t = pose
        depth = self.render_depth_map_mesh(K, R, t, self.height, self.width)

        tgt_dm, tgt_K, tgt_R, tgt_t = depth, K, R, t
        
        src_dms = self.src_dms
        src_Ks = self.src_Ks
        src_Rs = self.src_Rs
        src_ts = self.src_ts

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

        patch = np.array((0, tgt_dm.shape[0], 0, tgt_dm.shape[1]), dtype=np.int32)
        sampling_maps, valid_depth_masks, valid_map_masks = ext.preprocess.get_sampling_map(
            tgt_dm,
            tgt_K,
            tgt_R,
            tgt_t,
            src_dms,
            src_Ks,
            src_Rs,
            src_ts,
            patch,
            self.bwd_depth_thresh,
            self.invalid_depth_to_inf,
        )

        # make data ready for network to infer
        data = {}
        data["tgt_dm"] = tgt_dm
        data["sampling_maps"] = sampling_maps
        data["valid_depth_masks"] = valid_depth_masks
        data["valid_map_masks"] = valid_map_masks

        nbs = np.argsort(count)[::-1]
        nbs = nbs[: self.n_nbs]

        srcs = np.array([self.pad(load(self.src_im_paths[ii])) for ii in nbs])
        data["srcs"] = srcs

        tgt_height = min(tgt_dm.shape[0], patch[1]) - patch[0]
        tgt_width = min(tgt_dm.shape[1], patch[3]) - patch[2]
        data["tgt"] = np.zeros((3, tgt_height, tgt_width), dtype=np.float32)

        # network inference
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.net = self.net.to(self.eval_device)
            self.net.eval()
        image = self.net(**data)

        return image, count

    
    def get_reward(self, num_new_points, pose_change):
        """
        Calculate rewards based on number of new points and the cost for 
        """
        reward = num_new_points / self.num_total_points
        reward += pose_change * self.step_cost

        return reward


    def check_done(self):
        """
        Function to check if the current run is done
        """
        pass


    def render_depth_map_mesh(
        self,
        K,
        R,
        t,
        height,
        width,
        znear=0.05,
        zfar=1500,
        write_vis=False,
    ):

        scene = pyrender.Scene()
        mesh = pyrender.Mesh(
            primitives=[
                pyrender.Primitive(
                    positions=verts,
                    normals=normals,
                    color_0=colors,
                    indices=faces,
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

        if write_vis:
            depth[depth <= 0] = np.NaN
            depth = co.plt.image_colorcode(depth)
            imwrite(dm_path.with_suffix(".jpg"), depth)

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