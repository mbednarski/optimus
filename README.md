# KQV


Ala ma kota
Alice has a cat

Ala ma kota
Ala ma kota
 
Q(Ala) Q(ma) Q(kota)
K(Ala) K(ma) K(kota)

Q(Ala) K(Ala) = 0.1
Q(Ala) K(ma) = 0.3
Q(Ala) K(kota) = 0.6

Q(ma) K(Ala) = 0.1
Q(ma) K(ma) = 0.3
Q(ma) K(kota) = 0.6
---
Q(kota) K(Ala) = 0.1
Q(kota) K(ma) = 0.3
Q(kota) K(kota) = 0.6

V(Ala) V(ma) V(kota)

attn(Kota, ala) = 0.1 * V(Ala) + 0.3*V(ma) + 0.6*V(kota)
//attn(Kota, ala) = Q(kota)K(Ala)*V(Ala) + Q(kota)K(ma)*V(ma) + Q(kota)K(kota)*V(kota)

n=3
n_atnn=9 = n^2


attn(Q,K) * V
attn(Q,K,V)