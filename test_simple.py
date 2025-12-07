#!/usr/bin/env python
"""간단한 테스트 - Hand 클래스 초기화만 확인"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("1. Importing libraries...")
from demo import Hand

print("2. Initializing Hand...")
hand = Hand()

print("3. Generating simple handwriting...")
lines = ["Hello World"]
biases = [0.75]
styles = [9]

hand.write(
    filename='img/test_output.svg',
    lines=lines,
    biases=biases,
    styles=styles
)

print("4. Done! Check img/test_output.svg")
