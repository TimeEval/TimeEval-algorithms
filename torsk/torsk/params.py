import json
import pathlib
from marshmallow import Schema, fields, validate


_MODULE_DIR = pathlib.Path(__file__).parent.absolute()


class InputMap(Schema):
    type = fields.String(
        validate=validate.OneOf(
            ["pixels", "dct", "conv", "random_weights", "gradient", "compose"]),
        required=True)

    input_scale = fields.Float()
    operations = fields.Nested("self", many=True)

    # xsize if pixels
    # ksize if dct
    # kernel_shape if conv
    # hidden_size if random_weights
    size = fields.List(fields.Int())

    kernel_type = fields.String(
        validate=validate.OneOf(["mean", "gauss", "random"]))
    mode = fields.String(
        default="same",
        validate=validate.OneOf(["same", "same"]))


class ParamsSchema(Schema):
    input_shape = fields.List(fields.Int(), required=True)
    input_map_specs = fields.List(fields.Nested(InputMap()), required=True)

    reservoir_representation = fields.String(
        validate=validate.OneOf(["sparse", "dense"]), required=True)
    spectral_radius = fields.Float(required=True)
    density = fields.Float(required=True)

    train_length = fields.Int(required=True)
    pred_length = fields.Int(required=True)
    transient_length = fields.Int(required=True)

    train_method = fields.String(
        validate=validate.OneOf(["pinv_svd", "pinv_lstsq", "tikhonov"]),
        required=True)
    tikhonov_beta = fields.Float(missing=None)
    imed_loss = fields.Boolean(required=True)
    imed_sigma = fields.Float(missing=None)

    backend = fields.String(
        validate=validate.OneOf(["numpy"]), required=True)
    dtype = fields.String(
        valdiate=validate.OneOf(["float32", "float64"]), required=True)

    debug = fields.Boolean(required=True)

    anomaly_start = fields.Int()
    anomaly_step = fields.Int()
    timing_depth = fields.Int()


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path=None, params=None):
        if json_path is not None and params is not None:
            raise ValueError("json_path and params are mutually exclusive args")

        schema = ParamsSchema()

        if json_path is not None:
            with open(str(json_path)) as f:
                self.__dict__ = schema.loads(f.read())

        if params is not None:
            self.__dict__ = schema.load(params)

    def save(self, json_path):
        with open(str(json_path), 'w') as f:
            dump = ParamsSchema().dump(self.__dict__)
            json.dump(dump, f, indent=4)

    def update(self, params):
        """Updates parameters based on a dictionary or a list."""
        if isinstance(params, list):
            for i in range(0, len(params), 2):
                key, value = params[i], params[i + 1]
                try:
                    value = eval(value)
                except Exception:
                    pass
                self.__dict__[key] = value
        else:
            self.__dict__.update(params)
            # self.__dict__ = ParamsSchema().load(self.__dict__)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by
        `params.dict['learning_rate']"""
        return self.__dict__

    def __str__(self):
        ps = self.__dict__.copy()
        maps = ps.pop("input_map_specs")

        def maps_table_row(imap):
            if 'input_scale' in imap:
                scale = f"{imap['input_scale']:.2f}"
            else:
                scale = " -- "

            if imap['type'] in ['pixels', 'dct', 'random_weights']:
                maps_row = f"  {imap['type']:<8} {scale:<3} {imap['size']}\n"
            elif imap['type'] == 'conv':
                maps_row = f"  {imap['type']:<8} {scale:<3} {str(imap['size']):<9} {imap['kernel_type']}\n"
            elif imap['type'] == 'gradient':
                maps_row = f"  {imap['type']:<8} {scale:<3}\n"
            elif imap['type'] == 'compose':
                ops = imap['operations']
                maps_row = "  compose\n"
                for m in ops:
                    maps_row += f"  {maps_table_row(m)}"
            return maps_row

        maps_table = ""
        for imap in maps:
            maps_table += maps_table_row(imap)

        ps_dump = json.dumps(ps, indent=4, sort_keys=True)
        return f"Params:\n{ps_dump}\nInput maps:\n{maps_table}"


def default_params():
    json_path = _MODULE_DIR / "default_params.json"
    return Params(json_path=json_path)
