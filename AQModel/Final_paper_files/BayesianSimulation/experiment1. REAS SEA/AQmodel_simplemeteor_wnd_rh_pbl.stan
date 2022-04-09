// based on mekong methane made by Arai a*ebx * c[0,1] *r(pseudo_in_eachplot)* r(water) * r(location)


data {         //csv fileの構成も、以下の情報に対応するように整理する。(N_sampleと、id名もすべきか？)
	int N; // サンプルサイズ
	int<lower=1,upper=16> C; // CountID of city.
	int<lower=1,upper=2> T; // Tier.
	
	real<lower=0> R[N]; // dependent variable: R corrected by humidity beta
	real<lower=0> AR[N]; // net residential fixed
	real<lower=0> AC[N]; // net commercial fixed
	real<lower=0> AI[N]; // net industrial fixed
	real<lower=0> AF[N]; // net crop fire fixed 
	real<lower=0> ABK[N]; // net BK fixed
	real<lower=0> AV[N]; // net vehicle fixed
	real<lower=0> wnd[N]; // wind  fixed
	// real<lower=0> tmax[N]; // monthly mena temp max fixed
	real<lower=0> rhum[N]; // monthly rhumidity fixed
	real<lower=0> hpbl[N]; // monthly planetary boundayr layer fixed
	// real vwnd[N]; // wind towards north fixed
	int<lower=1> cityID2[N]; // city  represented C random
	int<lower=1> C2T[C]; // CtoT random

}

	
parameters {
  real beta1_0;
  real<lower=0> beta2_0;
  real<lower=0> beta3_0;
  real<lower=0> beta4_0;
  real<lower=0> beta5_0;
  real<lower=0> beta6_0;
  real<lower=0> beta7_0;
  real<upper=0> beta8_0;
  real<lower=0> beta9_0;
  real<upper=0> beta10_0;
  
  
  real beta1_2[T];
  real<lower=0> beta2_2[T];
  real<lower=0> beta3_2[T];
  real<lower=0> beta4_2[T];
  real<lower=0> beta5_2[T];
  real<lower=0> beta6_2[T];
  real<lower=0> beta7_2[T];  
  real<upper=0> beta8_2[T];  
  real<lower=0> beta9_2[T];  
  real<upper=0> beta10_2[T];  
 
  real beta1[C];
  real<lower=0> beta2[C];
  real<lower=0> beta3[C];
  real<lower=0> beta4[C];
  real<lower=0> beta5[C];
  real<lower=0> beta6[C];
  real<lower=0> beta7[C];  
  real<upper=0> beta8[C];  
  real<lower=0> beta9[C];  
  real<upper=0> beta10[C];  
  
  real<lower=0> sigma_beta10r;
  real<lower=0> sigma_beta20;
  real<lower=0> sigma_beta30;
  real<lower=0> sigma_beta40;
  real<lower=0> sigma_beta50;
  real<lower=0> sigma_beta60;
  real<lower=0> sigma_beta70;
  real<lower=0> sigma_beta80;
  real<lower=0> sigma_beta90;
  real<lower=0> sigma_beta100;

  
  real<lower=0> sigma_beta1;
  real<lower=0> sigma_beta2;
  real<lower=0> sigma_beta3;
  real<lower=0> sigma_beta4;
  real<lower=0> sigma_beta5;
  real<lower=0> sigma_beta6;
  real<lower=0> sigma_beta7;
  real<lower=0> sigma_beta8;  
  real<lower=0> sigma_beta9;
  real<lower=0> sigma_beta10;

  real<lower=0> s_Y;
}

model {

  for (t in 1:T) {
  
    beta1_2[t] ~ normal(beta1_0, sigma_beta10r);
    beta2_2[t] ~ normal(beta2_0, sigma_beta20);
    beta3_2[t] ~ normal(beta3_0, sigma_beta30);
    beta4_2[t] ~ normal(beta4_0, sigma_beta40);
	beta5_2[t] ~ normal(beta5_0, sigma_beta50);
	beta6_2[t] ~ normal(beta6_0, sigma_beta60);
	beta7_2[t] ~ normal(beta7_0, sigma_beta70);
	beta8_2[t] ~ normal(beta8_0, sigma_beta80);
	beta9_2[t] ~ normal(beta9_0, sigma_beta90);
	beta10_2[t] ~ normal(beta10_0, sigma_beta100);
  
  }

  for (c in 1:C) {    //y, Y軸のデータと混同しないか？ここのyはあくまでもyear
    beta1[c] ~ normal(beta1_2[C2T[c]], sigma_beta1);
    beta2[c] ~ normal(beta2_2[C2T[c]], sigma_beta2);
    beta3[c] ~ normal(beta3_2[C2T[c]], sigma_beta3);
    beta4[c] ~ normal(beta4_2[C2T[c]], sigma_beta4);
	beta5[c] ~ normal(beta5_2[C2T[c]], sigma_beta5);
	beta6[c] ~ normal(beta6_2[C2T[c]], sigma_beta6);
	beta7[c] ~ normal(beta7_2[C2T[c]], sigma_beta7);
	beta8[c] ~ normal(beta8_2[C2T[c]], sigma_beta8);
	beta9[c] ~ normal(beta9_2[C2T[c]], sigma_beta9);
	beta10[c] ~ normal(beta10_2[C2T[c]], sigma_beta10);
	
  }

  for (n in 1:N)     
    R[n] ~ normal(beta1[cityID2[n]] + beta2[cityID2[n]]*AR[n] + beta3[cityID2[n]]*AC[n] + beta4[cityID2[n]]*AI[n] + beta5[cityID2[n]]*AF[n] + beta6[cityID2[n]]*ABK[n] + beta7[cityID2[n]]*AV[n] + beta8[cityID2[n]]*wnd[n] + beta9[cityID2[n]]*rhum[n] + beta10[cityID2[n]]*hpbl[n], s_Y); //     , s_Y);
}


