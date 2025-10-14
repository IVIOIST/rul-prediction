# WTF is Remaining Useful Life (RUL)????
## Prologue
Imagine you're an engineer working on a fleet of aircraft engines. Your job is to ensure that these engines operate safely and efficiently. One of the most critical aspects of your job is predicting when an engine might fail or require maintenance. This is where the concept of Remaining Useful Life (RUL) comes into play.

## What about in the Context of This Project?
In this project we're decoding (yes I mean it in the literal sense) the Remaining Useful Life (RUL) of aircraft engines using sensor data. The goal is to predict how many cycles (or flights) an engine can continue to operate before it needs maintenance or is likely to fail. However due to the nature of the testing and the training data, we had to make some interpretations.

### Training Data
The training data consists of a series of sensor readings taken from 100 engines over time. Each engine runs until it fails, and the data includes the number of cycles each engine has completed before failure. This means that for each engine in the training set, we know exactly how many cycles it lasted before it broke down. As such we are defining RUL in the training data as the number of cycles remaining until failure. For example, if an engine has completed 50 cycles and it failed at 100 cycles, its RUL at that point would be 50 cycles.

### Testing Data
The testing data is a bit different. It also consists of sensor readings from engines, but these engines have not yet failed. Instead, the testing data includes engines that are still operational, and we don't know when they will fail. The ground truth RUL is provided separately and represents the RUL of the engine at the last recorded cycle in the test data. For example, if an engine has completed 70 cycles and the ground truth RUL is 30, it means that the engine is expected to fail after 100 cycles (70 completed + 30 remaining).

### Key Differences
1. **Training Data**: RUL is calculated based on the total number of cycles until failure. It is a direct measure of how many cycles are left before the engine fails.
2. **Testing Data**: RUL is provided as a ground truth value that indicates how many cycles are left from the last recorded cycle in the test data. It is not calculated from the data itself but given as a reference.