#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#define numInputNodes  4
#define numLayer1Nodes 12 
#define numLayer2Nodes 12
#define numOutputNodes 3
#define numTraininginputs 100
#define totalInput 150

int getValueForFlower(char flower[]){
    if(flower[1] == 'S' || flower[1] =='s'){
        return 0;
    }
    if(flower[2]=='e'){
        return 1;
    }
    return 2;
}
double sigmoid(double x){
     return 1 / (1 + exp(-x));
    }
double dSigmoid(double x) { 
    return x * (1.00 - x); 
    }
double learningRate = 0.08;
double input[totalInput][numInputNodes];
double actualOutput[totalInput][numOutputNodes];
double correctAnswer[totalInput];

double inputLayer[numInputNodes];
double layer1[numLayer1Nodes];
double layer2[numLayer2Nodes];
double outputLayer[numOutputNodes];

double layer1Bias[numLayer1Nodes];
double layer2Bias[numLayer2Nodes];
double outputBias[numOutputNodes];

double inpLayer1Weights[numInputNodes][numLayer1Nodes];
double layer1Layer2Weights[numLayer1Nodes][numLayer2Nodes];
double layer2OutputWeights[numLayer2Nodes][numOutputNodes];
double errorsInLastLayer[numOutputNodes];

void getInputdata(){ // this function is for reading the csv file and converting into array..
                    //this will fill input and actual output arrays...
    FILE *file = fopen("iris1.csv","r");
    char flower[20];
    for(int i=0;i<totalInput;i++){
    fscanf(file,"%lf,%lf,%lf,%lf,%[^\n]%*c",&input[i][0],&input[i][1],&input[i][2],&input[i][3],flower);
    double res;
    // printf("row = %d %lf %lf %lf %lf %s\n" ,i, input[i][0] , input[i][1],input[i][2],input[i][3],flower);
    res = getValueForFlower(flower);
    for(int k = 0;k<numOutputNodes;k++){actualOutput[i][k] = 0;}
    actualOutput[i][(int)res] = 1;
    correctAnswer[i] = res;
    }
    fclose(file);
}

void randomlySet(){ //set intial weights and bias to random values
for(int j=0;j<numLayer1Nodes;j++){
layer1Bias[j] = (random()%5)/5;
for(int i=0;i<numInputNodes;i++){
        inpLayer1Weights[i][j] = ((double)rand())/((double)RAND_MAX);
    }
}
for(int j=0;j<numLayer2Nodes;j++){
layer2Bias[j] = (random()%5) /5;
for(int i=0;i<numLayer1Nodes;i++){
        layer1Layer2Weights[i][j] = ((double)rand())/((double)RAND_MAX);
    }
}

for(int j=0;j<numOutputNodes;j++){
    errorsInLastLayer[j] = 0;
    outputBias[j] = (random()%5)/5;
    for(int i=0;i<numLayer2Nodes;i++){
        layer2OutputWeights[i][j] = ((double)rand())/((double)RAND_MAX);
    }
}
}
void doForwardPropagation(int row){
    //for each row in training set.. do 
    // for(int row = 0;row<numTraininginputs;row++){
        //set activation nodes of input to be training row's    
        // printf("for row = %d\n",row);
        int i=row;
        // printf("row = %d %lf %lf %lf %lf\n" ,row, input[i][0] , input[i][1],input[i][2],input[i][3]);
        // printf("input layer : ");
        for(int i=0;i<numInputNodes;i++){
            inputLayer[i] = input[row][i];
            // printf("%lf ",inputLayer[i]);
        }     
        // printf("\nl1 : ");
        //hiddenLayer 1
        for(int j = 0;j<numLayer1Nodes;j++){
            //initial value will be 0;
            layer1[j] = 0;
            for(int i=0;i<numInputNodes;i++){
                layer1[j] += (inpLayer1Weights[i][j] *inputLayer[i]); 
            }
            layer1[j] = sigmoid(layer1[j]);
            
        }
        //hiddenLayer 2
        // printf("\nl2: ");
        for(int j=0;j<numLayer2Nodes;j++){
            layer2[j] = 0;
            for(int i=0;i<numLayer1Nodes;i++){
                layer2[j] += (layer1[i] * layer1Layer2Weights[i][j]);
            }
            layer2[j] = sigmoid(layer2[j]);
            // printf("%lf ",layer2[j]);
        }
        printf("\noutput: for row = %d ",row);
        //now hiddenLayer 2  to output layer..
        for(int j=0;j<numOutputNodes;j++){
            outputLayer[j] = 0;
            for(int i=0;i<numLayer2Nodes;i++){
                outputLayer[j] += (layer2[i] * layer2OutputWeights[i][j]);
                // printf("%lf * %lf\n",layer2[i] , layer2OutputWeights[i][j]);
            }
            outputLayer[j] = sigmoid(outputLayer[j]);
            printf("%lf  ",outputLayer[j]);
        } 
        printf("\n");
}

void doBackPropagation(int row){ // with the values in errorsInLastLayer
                        // we will find the weight adjustment..
    // first compute the value of delta of output layer..
    // intial_weights
    double deltaOutput[numOutputNodes];
    for(int j=0;j<numOutputNodes;j++){
    deltaOutput[j] = (actualOutput[row][j] - outputLayer[j]) * dSigmoid(outputLayer[j]);
    }
    double deltaLayer2[numLayer2Nodes];
    for(int j=0;j<numLayer2Nodes;j++){
        double curr = 0;
        for(int k=0;k<numOutputNodes;k++){
            curr += (deltaOutput[k]*layer2OutputWeights[j][k]);
        }
        deltaLayer2[j] = curr * dSigmoid(layer2[j]);
    }
    double deltaLayer1[numLayer1Nodes];
    for(int j=0;j<numLayer1Nodes;j++){
        double curr = 0;
        for(int k=0;k<numLayer2Nodes;k++){
            curr += (deltaLayer2[k] * layer1Layer2Weights[j][k]);
        }
        deltaLayer1[j] = curr * dSigmoid(layer1[j]);
    }
    // now using those deltas change the weights..
    for(int j=0;j<numOutputNodes;j++){
        outputBias[j] += (learningRate * deltaOutput[j]);
        for(int k=0;k<numLayer2Nodes;k++){
            layer2OutputWeights[k][j] += (layer2[k]*deltaOutput[j] * learningRate);
        }
    }
    for(int j=0;j<numLayer2Nodes;j++){
        layer2Bias[j] += (learningRate * deltaLayer2[j]);
        for(int k=0;k<numLayer1Nodes;k++){
            layer1Layer2Weights[k][j] += (layer1[k] * deltaLayer2[j] * learningRate );
        }
    }
    for(int j=0;j<numLayer1Nodes;j++){
        layer1Bias[j] += (learningRate * deltaLayer1[j]);
        for(int k=0;k<numInputNodes;k++){
            inpLayer1Weights[k][j] += ( input[row][k] *deltaLayer1[j] *learningRate);
        }
    }
}

void trainModel(int times){
    // doForwardPropagation(0);
    // doBackPropagation(0);
    for(int t = 0;t<times;t++){
        for(int i=0;i<3;i++){
            for(int j = 0;j<30;j++){
                doForwardPropagation(i*50 + j);
                doBackPropagation(i*50 + j);
            }
        }
    }
}

double doTesting(){
    double correctTested = 0,totalTested = 0;
    for(int j=0;j<3;j++){
        for(int k=30;k<50;k++){
        int i = k+ j*50;
        doForwardPropagation(i);
         // output layer has our model outputs..
        totalTested++;
        int c=0;
        int maxi =0;
        for(int k = 1;k<numOutputNodes;k++){ //lets compare output and actual output
            if(outputLayer[maxi] <outputLayer[k]){
                maxi = k;
            }
        }
        printf("%lf %d\n",correctAnswer[i] , maxi);
        if(correctAnswer[i] == (double)maxi){
            correctTested++;
        }
    }
    }
    return correctTested/totalTested;
}

int main(){
    double a,b,c,d;
    getInputdata(); //after filling input data of size numTraininginputs
    randomlySet();
    trainModel(10000);
    //after training the model now test it with the remaining data...
    double accuracy = doTesting();
    printf("%lf is accuracy\n",accuracy);
    return 0;
}
