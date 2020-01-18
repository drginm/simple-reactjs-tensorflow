import React, { useState } from 'react';
import update from 'immutability-helper';
import * as tf from '@tensorflow/tfjs';

import './TensorflowExample.css';

const TensorflowExample = () => {
    //Value pairs state
    const [valuePairsState, setValuePairsState] = useState([
        { x: -1, y: -3 },
        { x: 0, y: -1 },
        { x: 1, y: 1 },
        { x: 2, y: 3 },
        { x: 3, y: 5 },
        { x: 4, y: 7 },
    ]);

    //Define the model state
    const [modelState, setModelState] = useState({
        model: null,
        trained: false,
        predictedValue: 'Click on train!',
        valueToPredict: 1,
    });

    //Event handlers
    const handleValuePairChange = (e) => {
        const updatedValuePairs = update(valuePairsState, {
            [e.target.dataset.index]: {
                [e.target.name]: { $set: parseInt(e.target.value) }
            }
        })

        setValuePairsState(
            updatedValuePairs
        )
    };

    const handleAddItem = () => {
        setValuePairsState([
            ...valuePairsState,
            { x: 1, y: 1 }
        ]);
    };

    const handleModelChange = (e) => setModelState({
        ...modelState,
        [e.target.name]: [parseInt(e.target.value)],
    });

    const handleTrainModel = () => {
        let xValues = [],
            yValues = [];

        valuePairsState.forEach((val, index) => {
            xValues.push(val.x);
            yValues.push(val.y);
        });

        // Define a model for linear regression.
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

        // Prepare the model for training: Specify the loss and the optimizer.
        model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
        const xs = tf.tensor2d(xValues, [xValues.length, 1]);
        const ys = tf.tensor2d(yValues, [yValues.length, 1]);

        // Train the model using the data.
        model.fit(xs, ys, { epochs: 250 }).then(() => {
            setModelState({
                ...modelState,
                model: model,
                trained: true,
                predictedValue: 'Ready for making predictions',
            });
        });
    }

    const handlePredict = () => {
        // Use the model to do inference on a data point the model hasn't seen before:
        const predictedValue = modelState.model.predict(tf.tensor2d([modelState.valueToPredict], [1, 1])).arraySync()[0][0];

        setModelState({
            ...modelState,
            predictedValue: predictedValue,
        });
    }

    return (
        <div className="tensorflow-example">
            <div className="train-controls">
                <h2 className="section">Training Data (x,y) pairs</h2>
                <div className="row labels">
                    <div className="field-label column">X</div>
                    <div className="field-label column">Y</div>
                </div>

                {valuePairsState.map((val, index) => {
                    return (
                        <div key={index} className="row">
                            <input
                                className="field field-x column"
                                value={val.x}
                                name="x"
                                data-index={index}
                                onChange={handleValuePairChange}
                                type="number" pattern="[0-9]*" />
                            <input
                                className="field field-y column"
                                value={val.y}
                                name="y"
                                data-index={index}
                                onChange={handleValuePairChange}
                                type="number" />
                        </div>
                    );
                })}

                <button
                    className="button-add-example button--green"
                    onClick={handleAddItem}>
                    +
                </button>
                <button
                    className="button-train button--green"
                    onClick={handleTrainModel}>
                    Train
                </button>
            </div>

            <div className="predict-controls">
                <h2 className="section">Predicting</h2>
                <input
                    className="field element"
                    value={modelState.valueToPredict}
                    name="valueToPredict"
                    onChange={handleModelChange}
                    type="number"
                    placeholder="Enter an integer number" /><br />
                <div className="element">
                    {modelState.predictedValue}
                </div>
                <button
                    className="element button--green"
                    onClick={handlePredict}
                    disabled={!modelState.trained}>
                    Predict
                </button>
            </div>
        </div>
    );
};

export default TensorflowExample;
