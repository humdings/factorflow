{
  description = "factorflow: Pipeline-style factor engine on a bar grid (batch + incremental open-bar execution)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      # Build factorflow with any Python package set (used by the overlay and packages).
      mkFactorflow = ps:
        ps.buildPythonPackage {
          pname = "factorflow";
          version = "0.0.1";
          pyproject = true;
          src = self;
          build-system = [ ps.hatchling ];
          dependencies = [ ps.numpy ps.polars ps.exchange-calendars ];
          nativeCheckInputs = [ ps.pytestCheckHook ps.hypothesis ];
          # Property tests are slow-ish; keep them in CI, run the fast unit tests in the build.
          pytestFlagsArray = [ "-m" "not slow" ];
          pythonImportsCheck = [ "factorflow" ];
        };
    in
    {
      # humdesk: inputs.factorflow.url = "github:humdings/factorflow";
      #          nixpkgs.overlays = [ inputs.factorflow.overlays.default ];
      #          then python3.pkgs.factorflow is available.
      overlays.default = final: prev: {
        pythonPackagesExtensions = (prev.pythonPackagesExtensions or [ ]) ++ [
          (pyfinal: pyprev: { factorflow = mkFactorflow pyfinal; })
        ];
      };
    }
    // flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python313;
        factorflow = mkFactorflow python.pkgs;
      in
      {
        packages.default = factorflow;
        packages.factorflow = factorflow;
        checks.default = factorflow;
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.uv python pkgs.ruff ];
          shellHook = ''
            export UV_PYTHON=${python}/bin/python
          '';
        };
      });
}
