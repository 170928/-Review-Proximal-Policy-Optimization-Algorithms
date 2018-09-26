# -Review-Proximal-Policy-Optimization-Algorithms
[Review]
> John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov  
> OpenAI  
> https://arxiv.org/pdf/1707.06347.pdf  
[Reference for git]  
> https://medium.com/@jonathan_hui/rl-the-math-behind-trpo-ppo-d12f6c745f33

## [Mathematics]
> 이 repository에서는 해당 논문에서 사용되는 수학적인 함수들에 대해서 공부하고 적으려고 합니다.  
> 추후에 GAIL 과 TRPO 에서도 근간이 되는 수식들이므로 PPO 논문의 이해는 매우 중요합니다.  

기본적으로 Reinforcement Learning은 discounted expected reward를 최대화 하는 것을 목표로 합니다.  
이 expected discounted reward를 η (에타) 로 다음과 같이 표현합니다.  

![image](https://user-images.githubusercontent.com/40893452/46075159-986f9380-c1c4-11e8-8cd5-48616389ce29.png)

### MM Algorithm

PPO 와 TRPO 는 Minorize-Maximization (MM) 알고리즘을 기반으로 수식이 구성 됩니다.  
이 알고리즘은 "iterative method"를 기본으로 합니다. 
MM 알고리즘에서는 매 iteration 마다 위 그림의 "파란색 선"으로 표현되는 surrogate function M 을 찾는 것을 목표로 합니다.  

이 "surrogate function"이 갖는 의미는 다음과 같습니다.  
(1) expected discounted reward η (에타) 의 "lower bound function"  
(2) 현재 가지고 있는 정책 (policy)를 따라 근사한 η (에타)   
(3) 최적화 (optimize) 하기 쉬운 함수  
> optimize가 쉬운 이유는 이 surrogate function을 quadratic equation으로 근사할 것이기 때문 ...  

매 MM 알고리즘의 iteration 마다 optimal point M 을 찾습니다.  
그리고, point M을 "현재 사용할 정책 (policy)"로 사용합니다.  
![image](https://user-images.githubusercontent.com/40893452/46075518-a5d94d80-c1c5-11e8-86fa-47498fa061f8.png)

얻게 된 M 정책을 기반으로 lower bound를 다시 계산 하며 이 과정을 계속 반복하는 것이 MM 알고리즘 입니다.  
MM 알고리즘을 반복적으로 수행하는 것으로써 정책 (policy)를 지속적으로 향상되어 갑니다.  

### Objective Function

![image](https://user-images.githubusercontent.com/40893452/46075612-fea8e600-c1c5-11e8-9e1c-625051e8234c.png)

위의 objective function은 다음과 같이 해석할 수 있습니다.  
(1) adavantage function (*expected reward minus baseline to reduction variance*)을 최대화 하는 것이 목적입니다.  
(2) 업데이트 하는 새 정책 (policy)가 학습 이전의 정책 (old policy)로부터 너무 크게 변화하지 않도록 (not too different) 제한 합니다.  

수식에서 사용되는 notation들은 다음과 같이 정의 되며, 일반적인 강화학습 논문에서 사용되어 오던 개념이 그대로 적용됩니다.  
![image](https://user-images.githubusercontent.com/40893452/46076239-efc33300-c1c7-11e8-9702-24678c598ac0.png)

advantage의 수식을 통해서 우리는 2가지의 다른 정책 (policy)를 사용해서 한쪽의 policy의 reward를 계산할 수 있게 됩니다.  

![image](https://user-images.githubusercontent.com/40893452/46076881-e0dd8000-c1c9-11e8-8cb8-f073f740ff3f.png)

위의 과정을 통해서 현재 trajectories를 만드는 phi' 정책과 baseline을 구성하는 phi 정책간의 관계를 볼 수 있습니다.  
최종 결과에서 η(phi)를 우변으로 넘겨주면 다음과 같은 수식을 얻을 수 있습니다.  

![image](https://user-images.githubusercontent.com/40893452/46076907-f3f05000-c1c9-11e8-8522-3ae2427598e8.png)

expectation (기댓값) advantage는 우변의 sigma_s p(s) * sigma_a phi(a|s) * A(s, a) 로 변환 될 수 있습니다.  
앞의 두 sigma로 묶인 부분들은 확률이며, 해당 확률에서 얻을 수 있는 값을 A(s, a)로 보면  
흔히 이해할 수 있는 expectation에 대한 수식이 구성됩니다.  

### Function 𝓛

MM 알고리즘을 통해서 우리는 현재 정책 (current policy)에서 η (에타) expected discounted reward를 근사하는 것으로 lower bound를 찾고자 합니다.  
![image](https://user-images.githubusercontent.com/40893452/46077127-bdff9b80-c1ca-11e8-99c9-b2f149c77160.png)

그럼 function L은 function M의 lower bound 중 일부가 됩니다.  

![image](https://user-images.githubusercontent.com/40893452/46077159-db346a00-c1ca-11e8-936c-0e5264bf066b.png)


### Surrogate Function


## [Motivation]
 
