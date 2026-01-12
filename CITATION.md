# Citing NetSmith

## NetSmith

If you use NetSmith in your research, please cite:

```bibtex
@software{netsmith,
  title = {NetSmith: Fast Network Analysis Library},
  author = {Jones, K.},
  year = {2025},
  url = {https://github.com/kylejones200/netsmith},
  note = {Network analysis library with Rust acceleration}
}
```

## Original R ts2net Package

The **multivariate functionality** in this Python package is directly inspired by and implements the API from Leonardo N. Ferreira's original R package. Please also cite:

```bibtex
@article{ferreira2024,
  author = {Ferreira, Leonardo N.},
  title = {From time series to networks in R with the ts2net package},
  journal = {Applied Network Science},
  year = {2024},
  volume = {9},
  number = {1},
  pages = {32},
  doi = {10.1007/s41109-024-00642-2},
  url = {https://doi.org/10.1007/s41109-024-00642-2}
}
```

**R Package:**
- GitHub: https://github.com/lnferreira/ts2net
- CRAN: https://cran.r-project.org/package=ts2net

## Acknowledgments

NetSmith builds on established algorithms and implementations from the complex networks and network analysis literature:

- **Visibility Graphs**: Lacasa et al. (2008)
- **Recurrence Networks**: Marwan et al. (2009)
- **Transition Networks**: Zhang & Small (2006)
- **False Nearest Neighbors**: Kennel et al. (1992)

## Key References

### Visibility Graphs
```bibtex
@article{lacasa2008,
  title={From time series to complex networks: The visibility graph},
  author={Lacasa, Lucas and Luque, Bartolo and Ballesteros, Fernando and Luque, Jordi and Nuno, Juan Carlos},
  journal={Proceedings of the National Academy of Sciences},
  volume={105},
  number={13},
  pages={4972--4975},
  year={2008}
}
```

### Recurrence Networks
```bibtex
@article{marwan2009,
  title={Complex network approach for recurrence analysis of time series},
  author={Marwan, Norbert and Donges, Jonathan F and Zou, Yong and Donner, Reik V and Kurths, J{\"u}rgen},
  journal={Physics Letters A},
  volume={373},
  number={46},
  pages={4246--4254},
  year={2009}
}
```

### Ordinal Patterns (Transition Networks)
```bibtex
@article{bandt2002,
  title={Permutation entropy: a natural complexity measure for time series},
  author={Bandt, Christoph and Pompe, Bernd},
  journal={Physical Review Letters},
  volume={88},
  number={17},
  pages={174102},
  year={2002}
}
```

### False Nearest Neighbors
```bibtex
@article{kennel1992,
  title={Determining embedding dimension for phase-space reconstruction using a geometrical construction},
  author={Kennel, Matthew B and Brown, Reggie and Abarbanel, Henry DI},
  journal={Physical Review A},
  volume={45},
  number={6},
  pages={3403},
  year={1992}
}
```

## License

This Python package is released under the MIT License, maintaining compatibility with the original R package (also MIT License).

## Contact

**NetSmith:**
- GitHub Issues: [Report bugs or request features]

**Original R ts2net Package (referenced in historical context):**
- Author: Leonardo N. Ferreira
- Email: ferreira@leonardonascimento.com
- Website: https://leonardoferreira.com
- GitHub: https://github.com/lnferreira/ts2net

