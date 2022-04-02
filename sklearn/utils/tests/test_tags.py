import pytest

from sklearn.base import BaseEstimator
from sklearn.utils._tags import (
    _DEFAULT_TAGS,
    _safe_tags,
)


class NoTagsEstimator:
    pass


class MoreTagsEstimator:
    def __sklearn_tags__(self):
        return {"allow_nan": True}


@pytest.mark.parametrize(
    "estimator, err_msg",
    [
        (BaseEstimator(), "The key xxx is not defined in __sklearn_tags__"),
        (NoTagsEstimator(), "The key xxx is not defined in _DEFAULT_TAGS"),
    ],
)
def test_safe_tags_error(estimator, err_msg):
    # Check that safe_tags raises error in ambiguous case.
    with pytest.raises(ValueError, match=err_msg):
        _safe_tags(estimator, key="xxx")


@pytest.mark.parametrize(
    "estimator, key, expected_results",
    [
        (NoTagsEstimator(), None, _DEFAULT_TAGS),
        (NoTagsEstimator(), "allow_nan", _DEFAULT_TAGS["allow_nan"]),
        (MoreTagsEstimator(), None, {**_DEFAULT_TAGS, **{"allow_nan": True}}),
        (MoreTagsEstimator(), "allow_nan", True),
        (BaseEstimator(), None, _DEFAULT_TAGS),
        (BaseEstimator(), "allow_nan", _DEFAULT_TAGS["allow_nan"]),
        (BaseEstimator(), "allow_nan", _DEFAULT_TAGS["allow_nan"]),
    ],
)
def test_safe_tags_no_get_tags(estimator, key, expected_results):
    # check the behaviour of _safe_tags when an estimator does not implement
    # __sklearn_tags__
    assert _safe_tags(estimator, key=key) == expected_results

### Testing __sklearn_tags__() functionality

def test_sklearn_tag1():
    # simple test with adding one parameter to subclass of BaseEstimator 
    newtag = "requires_y"
    newtagval = True
    class ACustomEstimator(BaseEstimator):
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags[newtag] = newtagval
            return tags
    testtags = ACustomEstimator().__sklearn_tags__()
    assert newtag in testtags and newtagval == testtags[newtag]

def test_sklearn_tag_multi():
    # test with adding multiple parameters to subclass of BaseEstimator 
    newtag1 = "requires_y"
    newtagval1 = True
    newtag2 = "requires_z"
    newtagval2 = True
    class ACustomEstimator(BaseEstimator):
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags[newtag1] = newtagval1
            tags[newtag2] = newtagval2        
            return tags
    testtags = ACustomEstimator().__sklearn_tags__()
    flag = newtag1 in testtags and newtagval1 == testtags[newtag1]
    flag and newtag2 in testtags and newtagval2 == testtags[newtag2]

    assert flag

def test_sklearn_tag_extendtags():
    # test with extending tags of a subclass of BaseEstimator 
    newtag1 = "requires_y"
    newtagval1 = True
    newtagval2 = True
    class ACustomEstimator(BaseEstimator):
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags[newtag1] = [newtagval1]
            return tags
    class ACustomEstimator2(ACustomEstimator):
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags[newtag1].append(newtagval2)
            return tags            
    testtags = ACustomEstimator2().__sklearn_tags__()
    flag = newtag1 in testtags and newtagval1 in testtags[newtag1]
    flag and newtagval2 in testtags[newtag1]

    assert flag

def test_sklearn_tag_nestedInheritance():
    # check if tags exist in subclasses of the Mixins 
    newtag1 = "requires_y"
    newtagval1 = True
    newtag2 = "requires_z"
    newtagval2 = True
    class ACustomMixin1(BaseEstimator):
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags[newtag1] = newtagval1
            return tags
    class ACustomMixin2(ACustomMixin1):
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags[newtag2] = newtagval2
            return tags
    testtags = ACustomMixin2().__sklearn_tags__()
    flag = newtag1 in testtags and newtagval1 == testtags[newtag1]
    flag and newtag2 in testtags and newtagval2 == testtags[newtag2]

    assert flag

def test_sklearn_tag_multiInheritance():
    # check if tags exist in with multiple inherited classes
    newtag1 = "requires_y"
    newtagval1 = True
    newtag2 = "requires_z"
    newtagval2 = True
    class ACustomMixin1:
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags[newtag1] = newtagval1
            return tags
    class ACustomMixin2:
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags[newtag2] = newtagval2
            return tags
    class LotsOfMixins(ACustomMixin1,ACustomMixin2,BaseEstimator):
        pass
    testtags = LotsOfMixins().__sklearn_tags__()
    flag = newtag1 in testtags and newtagval1 == testtags[newtag1]
    flag and newtag2 in testtags and newtagval2 == testtags[newtag2]

    assert flag


def test_sklearn_tag_overwrite():
    # test overwriting tags
    newtag1 = "requires_y"
    newtagval1 = True
    newtagval2 = False
    class ACustomMixin1(BaseEstimator):
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags[newtag1] = newtagval1
            return tags
    class ACustomMixin2(ACustomMixin1):
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags[newtag1] = newtagval2
            return tags
    testtags = ACustomMixin2().__sklearn_tags__()
    flag = newtag1 in testtags and newtagval2 == testtags[newtag1]
    assert flag

def test_safe_tag():
    # test _safe_tag with another non BaseEstimator
    newtag1 = "newtag"
    newtagval1 = True
    class NotUsingBaseEstimator:
        def __sklearn_tags__(self):
            return {newtag1: newtagval1}

    testtags = _safe_tags(NotUsingBaseEstimator())
    flag = newtag1 in testtags and newtagval1 == testtags[newtag1]
    assert flag

