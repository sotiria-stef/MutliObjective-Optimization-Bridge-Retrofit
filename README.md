# Abstract
Bridges, mainly exposed in a multiple hazard environment, are the most vulnerable component of the road network. Damage of critical bridge components (i.e., piers, bearings and abutments) may result in loss of bridge functionality after a hazard event and, therefore, the rapid decision for the most appropriate retrofit technique is crucial in order to limit the related direct and indirect losses in short time after the event. In line with the above, a holistic methodology is proposed herein for the selection of the optimum retrofit technique for bridge piers (i.e., reinforced concrete or FRP jackets) utilizing advanced analysis results, multi-objective optimization techniques and genetic algorithms to derive the retrofit measure’s properties in order to meet the performance, cost and sustainability criteria. In most cases proposed in the literature, the selection of the retrofit measure is based on the seismic assessment results and, in particular, the fragility curves of bridges retrofitted with various schemes (i.e., RC, steel, or FRP jackets) and varying properties. However, a component-specific selection of the optimum retrofit measure properties is considered herein, also accounting for the as-built properties of the bridge pier studied and the targeted performance, cost and CO2 emissions criteria. The source code developed for the application of the approach proposed is also provided (in Github). Since both the components and the criteria are parametrically defined within the code, it could be practically used for different case studies, investigating the effect of as-built properties, retrofit measure properties, and selection criteria on the results. The proposed methodology is indicatively applied to a case study bridge pier, estimating the optimal RC and FRP jacket properties for selected performance, cost, and sustainability criteria, comparatively assessing the results.


## Code
In order to execute the code, two functions developed. User choose which one is about to run in:

```python
if __name__ == '__main__':
```

## Results
### Jacket
![Jacket Results](./img/type1.bmp "Jacket Results")

### FRP
![FRP Results](./img/type2.bmp "FRP Results")