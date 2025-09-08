# Project Title

One Paragraph of the project description


## Getting Started

The setup should be as easy as following the next steps (in Linux):

```bash
# Skip if already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone git@github.com:Joel-Baptista/pidog_rl.git

cd pidog_rl

uv sync
```

Afterwards, you should have the repository with python virtual environment with the correct python packages and versions. 

### Prerequisites

Requirements for the software and other tools to build, test and push 
- [UV package and project manager](https://docs.astral.sh/uv/)

### Installing

A step by step series of examples that tell you how to get a development
environment running

Say what the step will be

    Give the example

And repeat

    until finished

End with an example of getting some data out of the system or using it
for a little demo

## Adding new parts to PiDog

If you want to add costum parts to PiDog, or simply improve on the meshes provided, follow this steps.

1. Place new meshes inside the "model" folder
2. Run ```./build/meshes.sh```
3. Edit the ```model/pidog.xml``` file to either replace existing geometries or add new ones, given the names inside ```model/meshes.xml```
4. Run ```uv run tests/sit.py``` to verify the changes

Tip: You can use Blender to manually reorient the mesh before adding it to MuJoCo, for a more visual and interactive mesh manipulation

## Running the tests

Explain how to run the automated tests for this system

### Sample Tests

Explain what these tests test and why

    Give an example

### Style test

Checks if the best practices and the right coding style has been used.

    Give an example

## Deployment

Add additional notes to deploy this on a live system

## Built With

  - [Contributor Covenant](https://www.contributor-covenant.org/) - Used
    for the Code of Conduct
  - [Creative Commons](https://creativecommons.org/) - Used to choose
    the license

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/PurpleBooth/a-good-readme-template/tags).

## Authors

  - **Billie Thompson** - *Provided README Template* -
    [PurpleBooth](https://github.com/PurpleBooth)

See also the list of
[contributors](https://github.com/PurpleBooth/a-good-readme-template/contributors)
who participated in this project.

## License

This project is licensed under the [CC0 1.0 Universal](LICENSE.md)
Creative Commons License - see the [LICENSE.md](LICENSE.md) file for
details

## Acknowledgments

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc
