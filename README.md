# DisTrans

This is the repository for "Addressing Heterogeneity in Federated Learning via Distributional Transformation".

DisTrans is a novel generic framework designed to improve federated learning(FL) under various level of data heterogeneity using a double-input-channel model structure and a carefully optimized offset.

<img width="770" alt="Screen Shot 2022-07-17 at 12 00 35" src="https://user-images.githubusercontent.com/68920888/179408847-14f30d47-8112-44e0-8ef2-77596b8a8ef3.png">

The global model in FL setting with DisTrans learns better features than with state-of-the-art (SOTA) methods and thus gets higher accuracy. 

<img width="770" alt="Screen Shot 2022-07-17 at 12 08 41" src="https://user-images.githubusercontent.com/68920888/179410767-4357b64d-5588-47a9-ac3c-bf7d02791559.png">

We perform extensive evaluation of DisTrans using five different image datasets and compare it against SOTA methods. DisTrans outperforms SOTA FL methods across various distributional settings of the clients' local data by 1\%--10\% with respect to testing accuracy. Moreover, our evaluation shows that \sys achieves 1\%--7\% higher testing accuracy than  data transformation (or augmentation), i.e., mixup and AdvProp.

For more details of our method and evaluations, please refer to Section 5 of the paper.

To run the code, for i.i.d. federated learning scenario, 


```
python hyperMACFed.py
```
for non-i.i.d federated learning scenario,

```
python MACFed.py
```

