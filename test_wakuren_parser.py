import unittest


class WakurenParserTests(unittest.TestCase):
    def test_list_style_rows(self) -> None:
        import app

        html = """
        <html><body>
          <table>
            <tr><th>買い目</th><th>オッズ</th></tr>
            <tr><td>1-2</td><td class="Odds">4.2</td></tr>
            <tr><td>5-5</td><td class="Odds">7.7</td></tr>
            <tr><td>3-8</td><td class="Odds">12.5</td></tr>
          </table>
        </body></html>
        """
        m = app.parse_wakuren_odds_from_html(html)
        self.assertEqual(m.get("1-2"), 4.2)
        self.assertEqual(m.get("5-5"), 7.7)
        self.assertEqual(m.get("3-8"), 12.5)

    def test_matrix_style_table(self) -> None:
        import app

        # header: 1..3, rows: 1..3 (diagonalは空)
        html = """
        <html><body>
          <table>
            <tr><th></th><th>1</th><th>2</th><th>3</th></tr>
            <tr><th>1</th><td class="Odds">1.8</td><td class="Odds">2.0</td><td class="Odds">3.1</td></tr>
            <tr><th>2</th><td class="Odds">2.0</td><td>-</td><td class="Odds">4.2</td></tr>
            <tr><th>3</th><td class="Odds">3.1</td><td class="Odds">4.2</td><td>-</td></tr>
          </table>
        </body></html>
        """
        m = app.parse_wakuren_odds_from_html(html)
        self.assertEqual(m.get("1-1"), 1.8)
        self.assertEqual(m.get("1-2"), 2.0)
        self.assertEqual(m.get("1-3"), 3.1)
        self.assertEqual(m.get("2-3"), 4.2)

    def test_embedded_json_fallback(self) -> None:
        import app

        html = """
        <html><body>
          <script>
            window.__DATA__ = {"kumiban":"1-2","odds":"4.2"};
            window.__DATA2__ = {"kumiban":"2-8","odds":"12.0"};
          </script>
        </body></html>
        """
        m = app.parse_wakuren_odds_from_html(html)
        self.assertEqual(m.get("1-2"), 4.2)
        self.assertEqual(m.get("2-8"), 12.0)


if __name__ == "__main__":
    unittest.main()
