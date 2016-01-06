# Changelog

## [1.2][2016-01-06]
### Changed
- `fit` can be safely called multiple times on the same model instance.
- `shrink_to_fit` removed, reducing dependency on C++11 stdlib.
- not calling `float` on the Cython version any more.
