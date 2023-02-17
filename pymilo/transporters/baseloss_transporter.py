# TODO
# sklearn doesn't have baseloss for glms in python 3.7.9
# should do the imports dimensionally dynamic.


# Handle python 3.5 issues.
glm_models = [
    'GammaRegressor',
    'PoissonRegressor',
    'TweedieRegressor'
]
legacy_version = False
try:
    from sklearn._loss.loss import BaseLoss
    # So the python version is >= 3.8
    from sklearn.linear_model._glm import GammaRegressor
    from sklearn.linear_model._glm import PoissonRegressor
    from sklearn.linear_model._glm import TweedieRegressor
except:
    # if all bypasses are true, then we either don't have TweedieRegression(3.5) or we have other kind of TweedieRegreesion(3.7)
    try:
        from sklearn._loss.glm_distribution import (
        ExponentialDispersionModel,
        TweedieDistribution,
        EDM_DISTRIBUTIONS,
        )
        from sklearn.linear_model._glm.link import (
        BaseLink,
        IdentityLink,
        LogLink,
        )
        from sklearn.linear_model._glm import GammaRegressor
        from sklearn.linear_model._glm import PoissonRegressor
        from sklearn.linear_model._glm import TweedieRegressor
        legacy_version = True
    except:
        # there is not glm models.
        legacy_version

print("LEGACY-VERSION: ",legacy_version)

from ..utils.util import check_str_in_iterable
from .transporter import AbstractTransporter
# Handling BaseLoss function in GLMs.
# BaseLoss function in Tweedie regression
# BaseLoss function in Poisson regression
# BaseLoss function in Gamma regression

class BaseLossTransporter(AbstractTransporter):
    def serialize(self, data, key, model_type):
        # bypass when it's not supported
        # special legacy mode.
        if(model_type in glm_models):
            if(not(legacy_version)):
                # Handling latest GLMs with Loss function of GLMs 
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
                   
            else:
                # it's legacy version of GLMs
                if(key == "_family_instance"):
                    if model_type == "TweedieRegressor":
                        data["_family_instance"] = {
                                "family": {
                                    'state': 'not-direct-serializable',
                                    'value': {
                                        'power': data["power"]
                                    }
                                }
                            }
                    elif model_type == "PoissonRegressor":
                        data["_family_instance"] = {
                            "family":{
                                'state': 'direct-serializable',
                                'value': "poisson"
                            }
                        }
                    elif model_type == "GammaRegressor":
                        data["_family_instance"] = {
                            "family":{
                                'state': 'direct-serializable',
                                'value': "gamma"
                            }
                        }
                    return data["_family_instance"]

                elif(key == "_link_instance"):
                    return "sklean-mirror-link"                        
                elif(key == "link"):
                    if(data[key] in ['auto', 'identity', 'log']):
                        data[key] = {
                            'state': 'direct-serializable',
                            'value': data[key]
                        }
                        return data[key]
                    else:
                        if(isinstance(data[key], LogLink)):
                            data[key] = {
                                'state': 'not-direct-serializable',
                                'value': { 
                                    'abstract-class': "LogLink"                                
                                }
                            }
                            return data[key]
                        
                        elif(isinstance(data[key], IdentityLink)):
                            data[key] = {
                                'state': 'not-direct-serializable',
                                'value': { 
                                    'abstract-class': "IdentityLink"                                
                                }
                            }
                            return data[key]
                        
                        else:
                            # isinstance(data[key], BaseLink) == True
                            data[key] = {
                                'state': 'not-direct-serializable',
                                'value': { 
                                    'abstract-class': "BaseLink"                                
                                }
                            }
                            return data[key]
                else:
                    return data[key]
        else:
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
        # bypass when it's not supported
        # special legacy mode.
        if(model_type in glm_models):
            if(not(legacy_version)):
                # latest GLMs or irrelavant models.
                content = data[key]
                if not (check_str_in_iterable(
                        "pymilo_glm_base_loss", content)):
                    return content
                return self.get_deserialized_base_loss(model_type, content)
            else:
                # it's legacy version of GLMs
                if(key == "_family_instance"):
                        # family field retrieval... 
                        family = data["_family_instance"]["family"]
                        if(family['state'] == 'direct-serializable'):
                            family = family['value']
                        else:
                            family = TweedieDistribution(power=family['value']['power'])

                        if(isinstance(family, ExponentialDispersionModel)):
                            return family
                        elif family in EDM_DISTRIBUTIONS:
                            return EDM_DISTRIBUTIONS[family]()
                        else:
                            raise ValueError(
                                "The family must be an instance of class"
                                " ExponentialDispersionModel or an element of"
                                " ['normal', 'poisson', 'gamma', 'inverse-gaussian']"
                                "; got (family={0})".format(family)
                            )
                elif(key == "link"):
                    if(data[key]['state'] == 'direct-serializable'):
                        data[key] = data[key]['value']
                        return data[key]
                    else:
                        innerMap = {
                            'LogLink': LogLink,
                            'IdentityLink': IdentityLink,
                            'BaseLink': BaseLink
                        }
                        return innerMap[data[key]['value']['abstract-class']]()
                elif(key == "_link_instance"):
                    # make sure it has been deserialized.
                    try:
                        data["link"]["state"]
                        data["link"] = self.deserialize(data,"link",model_type)
                    except:
                        # it has been serialized.
                        data["link"]
                    if isinstance(data["link"], BaseLink):
                            return data["link"]
                    else:
                        if data["link"] == "auto":
                            if isinstance(data["_family_instance"], TweedieDistribution):
                                if data["_family_instance"].power() <= 0:
                                    return IdentityLink()
                                if data["_family_instance"].power() >= 1:
                                    return LogLink()
                            else:
                                raise ValueError(
                                    "No default link known for the "
                                    "specified distribution family. Please "
                                    "set link manually, i.e. not to 'auto'; "
                                    "got (link='auto', family={})".format(data["family"])
                                )
                        elif data["link"] == "identity":
                            return IdentityLink()
                        elif data["link"] == "log":
                            return LogLink()
                        else:
                            raise ValueError(
                                "The link must be an instance of class Link or "
                                "an element of ['auto', 'identity', 'log']; "
                                "got (link={0})".format(data["link"])
                            )
                else:
                    return data[key]
        else:
            return data[key]


