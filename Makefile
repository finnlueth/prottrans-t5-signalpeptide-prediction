SHELL := /bin/bash

.DEFAULT_GOAL := help
.PHONY: help clean venv

# Miscilaneous commands
help: ## Print this help.
	@grep -E '^[0-9a-zA-Z%_-]+:.*## .*$$' $(firstword $(MAKEFILE_LIST)) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

clean: ## Clean any artfacts and build components from system. Does not delete any downloaded data.
	@echo Removing .venv... & rm -rf .venv

venv: ## Create virtual environment using poetry.
	@poetry config virtualenvs.in-project true
	@poetry install --with dev
	@poetry config virtualenvs.in-project false
