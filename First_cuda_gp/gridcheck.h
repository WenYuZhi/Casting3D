#include <stdio.h>
void gridcheck(float dx, float dy, float tao)
{
	float a;
	struct physicial_parameter
	{
		float prediction_pho;
		float prediction_Ce;
		float prediction_lamd;
	} steel;
	steel.prediction_lamd = 50;
	steel.prediction_Ce = 540;
	steel.prediction_pho = 7000;
	a = steel.prediction_lamd / (steel.prediction_Ce*steel.prediction_pho);
	if ((a*tao) / (dx*dx) > 0.5 && (a*tao) / (dy*dy) > 0.5)
	{
		printf("The diffenence equation may be stability \n");
		printf("The stability coefficient is (%f, %f)\n", (a*tao) / (dx*dx), (a*tao) / (dy*dy));
	}
	else
	{
		printf("\n The diffenence equation may be unstability \n");
		printf("The stability coefficient is (%f, %f)\n", (a*tao) / (dx*dx), (a*tao) / (dy*dy));
	}
}