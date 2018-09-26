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




### Surrogate Function


## [Motivation]
 
