#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys


METADATA = ['title', 'description', 'version', 'authors', 'language']
PARAMETER_METADATA = ['name', 'description']
PARAMETER_TYPES = ['string', 'int', 'float', 'boolean', 'columnName']
RE_PARAMETER_TYPES = '(' + '|'.join(PARAMETER_TYPES) + ')'
ALGORITHM_TYPES = ['preprocessor', 'detector', 'classifier']
INPUT_DIMENSIONALITIES = ['univariate', 'multivariate']
MODEL_INPUT_OPTIONS = ['required', 'optional', 'none']
BOOLEAN_VALUES = ['false', 'true']
LEARNING_TYPES = ['supervised', 'unsupervised', 'semi-supervised']


def validate(path: str):
    '''Validate the manifest at the given path and print errors to stdout.
    Return True if the manifest is valid, else False.'''
    is_valid = True

    with open(path, 'r') as file:
        data = file.read()
    manifest = json.loads(data)

    # Check Metadata
    for key in METADATA:
        if not check_if_key_exists(key=key, json=manifest):
            is_valid = False
        else:
            if type(manifest[key]) != str:
                print(f'{key} is not a string')
                is_valid = False

    # Check Algorithm Type
    if not check_if_key_exists(key='type', json=manifest):
        is_valid = False
    if manifest['type'].lower() not in ALGORITHM_TYPES:
        print(f'{manifest["type"]} is not a valid Algorithm Type')
        is_valid = False

    # Check Main File
    if not check_if_key_exists(key='mainFile', json=manifest):
        is_valid = False
    else:
        if type(manifest['mainFile']) != str:
            print('mainFile is not a string')
            is_valid = False

    # Check Input Dimensionalities
    if not check_if_key_exists(key='inputDimensionality', json=manifest):
        is_valid = False
    else:
        if manifest['inputDimensionality'].lower() not in INPUT_DIMENSIONALITIES:
            print(f'{manifest["inputDimensionality"]} is not a valid Input Dimensionality')
            is_valid = False

    # Check Learning Type
    if not check_if_key_exists(key='learningType', json=manifest):
        is_valid = False
    else:
        if manifest['learningType'].lower() not in LEARNING_TYPES:
            print(f'{manifest["learningType"]} is not a valid Learning Type')
            is_valid = False

    # Check (optional) Training Step
    if ('trainingStep' in manifest and
            not check_step(step=manifest['trainingStep'], step_name='trainingStep')):
        is_valid = False

    # Check Execution Step
    if not check_if_key_exists(key='executionStep', json=manifest):
        is_valid = False
    else:
        if not check_step(step=manifest['executionStep'], step_name='executionStep'):
            is_valid = False

    return is_valid


def check_step(step, step_name: str):
    step_is_valid = True
    # Check Model Input
    if not check_if_key_exists(key='modelInput', json=step, step_name=step_name):
        step_is_valid = False
    else:
        if step['modelInput'].lower() not in MODEL_INPUT_OPTIONS:
            print(f'{step["modelInput"]} is not a valid Model Input Option in {step_name}')
            step_is_valid = False

    # Check Parameters
    if not check_if_key_exists(key='parameters', json=step, step_name=step_name):
        step_is_valid = False
    else:
        for parameter in step['parameters']:
            if not check_parameter(parameter, step_name=step_name):
                step_is_valid = False

    return step_is_valid


def check_parameter(parameter, step_name: str):
    parameter_is_valid = True
    # Check Name and Description of parameter
    for metadata in PARAMETER_METADATA:
        if not check_if_key_exists(metadata, json=parameter,
                                   is_param=True, step_name=step_name):
            parameter_is_valid = False
        else:
            if type(parameter[metadata]) != str:
                print(f'{parameter[metadata]} should be a string')
                parameter_is_valid = False

    # Check Type
    if not check_if_key_exists('type', json=parameter, is_param=True, step_name=step_name):
        parameter_is_valid = False
    else:
        if (parameter['type'].lower() not in PARAMETER_TYPES and not
            re.match(f'list\[{RE_PARAMETER_TYPES}\]', parameter['type'], re.IGNORECASE) and not
            re.match(r'enum\[.*\]', parameter['type'], re.IGNORECASE)):
            print(f'{parameter["type"]} is not a valid parameter type')
            parameter_is_valid = False

    # Check Optional
    if not check_if_key_exists('optional', json=parameter, is_param=True, step_name=step_name):
        parameter_is_valid = False
    else:
        if parameter['optional'].lower() not in BOOLEAN_VALUES:
            print(f'{parameter["optional"]} should be a bool')
            parameter_is_valid = False

    return parameter_is_valid


def check_if_key_exists(key: str, json, step_name: str = None, is_param: bool = False):
    if key not in json:
        message = f'{key} is missing'
        if is_param and 'name' in json:
            message = message + ' in parameter ' + json['name']
        if step_name:
            message = message + ' in step ' + step_name
        print(message)
    return key in json


def main():
    description = 'Validate an algorithm manifest.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--path', metavar='path', type=str,
                        help='Path to the manifest.'
                        'If not given the manifest in the current directory is used.')
    args = parser.parse_args()
    manifest_path = args.path
    if not manifest_path:
        manifest_path = os.path.join(os.getcwd(), 'manifest.json')

    if not os.path.isfile(manifest_path):
        print(f'{manifest_path} does not exist.')
        sys.exit(1)

    if validate(path=manifest_path):
        print(f'{manifest_path} is valid!')
        sys.exit()
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
