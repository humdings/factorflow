{
  description = "factorflow: Pipeline-style factor engine on a bar grid (batch + incremental open-bar execution)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      # Build factorflow with any Python package set (used by the overlay and packages).
      # exchange-calendars (and its pyluach dependency) are not in nixpkgs; use the wheels pinned in uv.lock.
      wheel = ps: { pname, version, url, hash, dependencies ? [ ] }:
        ps.buildPythonPackage {
          inherit pname version dependencies;
          format = "wheel";
          src = builtins.fetchurl { inherit url; sha256 = hash; };
          doCheck = false;
        };
      calendarDeps = ps:
        let
          pyluach = wheel ps {
            pname = "pyluach";
            version = "2.3.0";
            url = "https://files.pythonhosted.org/packages/8a/c8/f96208ade3ca4c23b372497d0788bcf0f2e0ff4310e5ee693366bc33fdf0/pyluach-2.3.0-py3-none-any.whl";
            hash = "sha256:4497b731aef59508b079dbf5f00bc5bf4329ac45090a6cd37b5a83756f0e69ab";
          };
        in
        ps.exchange-calendars or (wheel ps {
          pname = "exchange-calendars";
          version = "4.13.2";
          url = "https://files.pythonhosted.org/packages/c8/4c/0469b40057bc9f8d9594dcc6024202626b981ae4b52dfcd304552e8e1c3a/exchange_calendars-4.13.2-py3-none-any.whl";
          hash = "sha256:fc5a2ad0d61b5c3a6539a3061cd4cbb55c59f4a903455cec7926e4b798919996";
          dependencies = [ ps.numpy ps.pandas ps.toolz ps.tzdata ps.korean-lunar-calendar pyluach ];
        });
      mkFactorflow = ps:
        ps.buildPythonPackage {
          pname = "factorflow";
          version = "0.0.1";
          pyproject = true;
          src = self;
          build-system = [ ps.hatchling ];
          dependencies = [ ps.numpy ps.polars ps.pandas (calendarDeps ps) ];
          nativeCheckInputs = [ ps.pytestCheckHook ps.hypothesis ];
          # Tests marked slow (long simulations) run in CI / `uv run pytest`, not in the build.
          disabledTestMarks = [ "slow" ];
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
