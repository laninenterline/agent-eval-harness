class Aehcli < Formula
  include Language::Python::Virtualenv

  desc "Framework-neutral, task-level evaluation and regression harness for AI agents"
  homepage "https://github.com/laninenterline/agent-eval-harness"
  url "https://files.pythonhosted.org/packages/c3/ad/7beeebb82df97ae369e5115b1c26aa77711ec5b5d8d8644cdc98d1d4accd/aehcli-0.1.0.tar.gz"
  sha256 "d185e6fb79cae7f6c517fd7d81e3b84cdfe6b073db4d5dca928e037ecc98df06"
  license "Apache-2.0"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "usage: aeh", shell_output("#{bin}/aeh --help")
    assert_match "deepseek-v4-pro", shell_output("#{bin}/aeh judges")
  end
end
