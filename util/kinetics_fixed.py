class PairedKineticsFixed(PairedKinetics):
    def __init__(
            self,
            root,
            max_distance=48,
            repeated_sampling=2,
            seed=42
    ):
        super().__init__(root, max_distance, repeated_sampling, seed)
        self.presampled_indices = {}
        # Presample multiple pairs of indices for all videos
        for idx in range(len(self.samples)):
            sample = os.path.join(self.root, self.samples[idx][1])
            try:
                vr = VideoReader(sample, num_threads=1, ctx=cpu(0))
            except Exception as e:
                print(f"Error loading video {sample}: {str(e)}")
                # Return a default/empty sample
                return torch.zeros(self.repeated_sampling, 3, 224, 224), \
                    torch.zeros(self.repeated_sampling, 3, 224, 224), 0
            seg_len = len(vr)
            least_frames_num = self.max_distance + 1

            # Sample repeated_sampling pairs of frames
            pairs = []
            for _ in range(self.repeated_sampling):
                if seg_len >= least_frames_num:
                    idx_cur = random.randint(0, seg_len - least_frames_num)
                    interval = random.randint(4, self.max_distance)
                    idx_fut = idx_cur + interval
                else:
                    indices = random.sample(range(seg_len), 2)
                    indices.sort()
                    idx_cur, idx_fut = indices
                pairs.append((idx_cur, idx_fut))

            self.presampled_indices[idx] = pairs

    def load_frames(self, vr, index, pair_idx):
        idx_cur, idx_fut = self.presampled_indices[index][pair_idx]
        frame_cur = vr[idx_cur].asnumpy()
        frame_fut = vr[idx_fut].asnumpy()
        return frame_cur, frame_fut

    def __getitem__(self, index):
        sample = os.path.join(self.root, self.samples[index][1])
        vr = VideoReader(sample, num_threads=1, ctx=cpu(0))

        # Load and transform each pair of presampled frames
        src_images = []
        tgt_images = []
        for pair_idx in range(self.repeated_sampling):
            src_image, tgt_image = self.load_frames(vr, index, pair_idx)
            src_transformed, tgt_transformed = self.transform(src_image, tgt_image)
            src_images.append(src_transformed)
            tgt_images.append(tgt_transformed)

        src_images = torch.stack(src_images, dim=0)
        tgt_images = torch.stack(tgt_images, dim=0)
        return src_images, tgt_images, 0