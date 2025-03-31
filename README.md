# AI Engineer Take-Home Assignment

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
   - [Core Functionality](#core-functionality)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
5. [Usage](#usage)
6. [License](#license)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system designed to help insurance brokers efficiently understand and compare carrier appetites for different types of risks. By leveraging Large Language Models (LLMs) and a vector database, the system enables accurate processing of carrier appetite guides and provides coherent answers to specific queries.

## Features

### Core Functionality

1. **Document Processing**
    - Ingests and processes carrier appetite guides.
    - Implements a chunking strategy for efficient data handling.
    - Stores document embeddings in a vector database for quick retrieval.

2. **Query System**
    - Offers a query interface (CLI or simple API).
    - Retrieves relevant information accurately based on user queries.
    - Uses LLMs to generate clear and coherent responses, even for complex comparisons and implicit information.

3. **Example Queries**
    - Which carriers provide coverage in Arizona?
    - Which carrier can write a premium of X USD or lower?
    - Which carrier can handle a business of type X?
    - Find carriers that write apartment buildings with limits over $10M.
    - Which carriers would consider a $20M TIV warehouse in Arizona?

## Project Structure

```
```

## Setup

### Prerequisites

- Python 3.12.3
- Virtual environment

### Installation
- Create .env file with these keys:
OPENAI_API_KEY

## Usage

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute by submitting issues or pull requests!

