class CreatedAttackedBatch(BaseCommand):

    def adversarial_pattern(self, model, image, label):
        image = tensorflow.cast(image, tensorflow.float32)
        with tensorflow.GradientTape() as tape:
            tape.watch(image)
            prediction = model(image)
            loss = tensorflow.keras.losses.MSE(label, prediction)
        
        gradient = tape.gradient(loss, image)
        signed_grad = tensorflow.sign(gradient)
        
        return signed_grad

    def plot_predictions(self, cfg, tub_paths, model_path, limit, model_type):
        '''
        Plot model predictions for angle and throttle against data from tubs.

        '''
        import matplotlib.pyplot as plt
        import pandas as pd
        from tensorflow.python.keras.models import load_model, Model
                
        model_path = os.path.expanduser(model_path)
        model = dk.utils.get_model_by_type(model_type, cfg)
        if model_type is None:
            model_type = cfg.DEFAULT_MODEL_TYPE
        model.load(model_path)

        model = load_model(model_path, compile=False)

        user_angles = []
        user_throttles = []
        pilot_angles = []
        pilot_throttles = []       

        from donkeycar.parts.tub_v2 import Tub
        from pathlib import Path
        import tarfile
        from donkeycar.pipeline.types import TubRecord

        base_path = Path(os.path.expanduser(tub_paths)).absolute().as_posix()
        tub = Tub(base_path)
        records = list(tub)
        records = records[:limit]
        bar = IncrementalBar('Inferencing', max=len(records))

        tub1 = tub

        tub2 = Tub(os.path.join(base_path, 'data_ccmd/'),
                   inputs=['cam/image_array', 
                            # 'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                            # 'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z',
                            'user/angle', 
                            # 'user/mode', 
                            'user/throttle'],
                   types=['image_array', 
                            # 'float', 'float', 'float',
                            # 'float', 'float', 'float',
                            'float', 
                            # 'str', 
                            'float'])

        new_records = {}
        for key, record in enumerate(tub1):
            new_records[key] = record

        for key, record in enumerate(tub1):
            t_record = TubRecord(config=cfg,
                                 base_path=tub1.base_path,
                                 underlying=record)
            img_arr = t_record.image(cached=False)
            # record['cam/image_array'] = img_arr
            record['user/angle'] = record['user/angle']
            # record['imu/acl_x'] = record['imu/acl_x']
            # record['imu/acl_y'] = record['imu/acl_y']
            # record['imu/acl_z'] = record['imu/acl_z']
            # record['imu/gyr_x'] = record['imu/gyr_x']
            # record['imu/gyr_y'] = record['imu/gyr_y']
            # record['imu/gyr_z'] = record['imu/gyr_z']
            # record['user/mode'] = record['user/mode']
            record['user/throttle'] = record['user/throttle']


            imagem = img_arr.reshape((1,) + img_arr.shape)
            label_to_pass = model.predict(imagem)
            perturbation = self.adversarial_pattern(model, imagem, label_to_pass).numpy()
            perturb = ((perturbation[0]*0.5 + 0.5)*255)-50
            adv_img = np.clip(img_arr + (perturb*0.1), 0, 255)
            adv_img = adv_img.astype(int)

            print("im about to set a terrible atack on the image")

            norm_arr = normalize_image(adv_img)
            record['cam/image_array'] = adv_img
            tub2.write_record(record)


    def parse_args(self, args):
        parser = argparse.ArgumentParser(prog='tubplot', usage='%(prog)s [options]')
        parser.add_argument('--tub', nargs='+', help='The tub to make plot from')
        parser.add_argument('--model', default=None, help='model for predictions')
        parser.add_argument('--limit', type=int, default=1000, help='how many records to process')
        parser.add_argument('--type', default=None, help='model type')
        parser.add_argument('--config', default='./config.py', help=HELP_CONFIG)
        parsed_args = parser.parse_args(args)
        return parsed_args

    def run(self, args):
        args = self.parse_args(args)
        args.tub = ','.join(args.tub)
        cfg = load_config(args.config)
        self.plot_predictions(cfg, args.tub, args.model, args.limit, args.type)