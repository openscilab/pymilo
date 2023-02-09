bypass = {
    'GammaRegressor': True,
    'PoissonRegressor': True,
    'TweedieRegressor': True
}
try:
    from sklearn._loss.loss import BaseLoss
    from sklearn.linear_model._glm import GammaRegressor
    bypass['GammaRegressor'] = False
    from sklearn.linear_model._glm import PoissonRegressor
    bypass['PoissonRegressor'] = False
    from sklearn.linear_model._glm import TweedieRegressor
    bypass['TweedieRegressor'] = False
    from ..utils.util import check_str_in_iterable
    from .transporter import AbstractTransporter
except:
    bypass
# Handling BaseLoss function in GLMs.
# BaseLoss function in Tweedie regression
# BaseLoss function in Poisson regression
# BaseLoss function in Gamma regression
print("BYPASS:", bypass)

class BaseLossTransporter(AbstractTransporter):
    def serialize(self, data, key, model_type):
        # bypass when it's not supported.
        if(model_type in bypass.keys()):
            if(bypass[model_type]):
                return data[key]
        # Handling special Loss function of GLMs.
        if isinstance(data[key], BaseLoss):
            if model_type == "TweedieRegressor":
                data[key] = {
                    "power": data["power"],
                    "link": data["link"],
                    "pymilo_glm_base_loss": True
                }
            elif model_type == "PoissonRegressor":
                data[key] = {
                    "pymilo_glm_base_loss": True
                    # nothing for now.
                }
            elif model_type == "GammaRegressor":
                data[key] = {
                    "pymilo_glm_base_loss": True
                    # nothing for now
                }
            else:
                # print("ERROR: NOT IMPLEMENTED YET")
                # TODO
                return data[key]

        return data[key]

    def get_deserialized_base_loss(self, model_type, content):
        if model_type == "TweedieRegressor":
            if not ("power" in content and "link" in content):
                return None  # TODO EXCEPTION HANDLING
            power, link = content["power"], content["link"]
            return TweedieRegressor(power=power, link=link)._get_loss()
        elif model_type == "PoissonRegressor":
            return PoissonRegressor()._get_loss()
        elif model_type == "GammaRegressor":
            return GammaRegressor()._get_loss()
        else:
            # print("NOT IMPLEMENTED YET")
            # TODO
            return content

    def deserialize(self, data, key, model_type):
        # bypass when it's not supported.
        if(model_type in bypass.keys()):
            if(bypass[model_type]):
                return data[key]
            
        content = data[key]
        if not (check_str_in_iterable(
                "pymilo_glm_base_loss", content)):
            return content
        return self.get_deserialized_base_loss(model_type, content)
