"""
Comprehensive test suite for NaturalGit.
"""
import unittest
from naturalgit.heuristic_engine import match_heuristic
from naturalgit.git_context import GitContext


class TestNaturalGit(unittest.TestCase):
    def setUp(self):
        self.ctx = GitContext()
        self.ctx.branch = "feature/test"
        self.ctx.is_repo = True

    def test_status_queries(self):
        for q in [
            "stato",
            "cosa ho modificato",
            "cosa ho toccato",
            "quali file sono cambiati",
            "situazione",
            "mostrami lo stato dei file",
        ]:
            res = match_heuristic(q, self.ctx)
            self.assertIsNotNone(res, f"Failed on: {q}")
            self.assertEqual(res.command, "git status")

    def test_diff_queries(self):
        res = match_heuristic("differenze", self.ctx)
        self.assertIsNotNone(res)
        self.assertEqual(res.command, "git diff")

        res_staged = match_heuristic("differenze in staging", self.ctx)
        self.assertIsNotNone(res_staged)
        self.assertEqual(res_staged.command, "git diff --staged")

        res_file = match_heuristic("differenze per il file src/app.py", self.ctx)
        self.assertIsNotNone(res_file)
        self.assertEqual(res_file.command, "git diff src/app.py")

    def test_staging_queries(self):
        res1 = match_heuristic("aggiungi tutto", self.ctx)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.command, "git add -A")

        res2 = match_heuristic("metti in staging il file main.py", self.ctx)
        self.assertIsNotNone(res2)
        self.assertEqual(res2.command, "git add main.py")

        res3 = match_heuristic("togli il file main.py dallo staging", self.ctx)
        self.assertIsNotNone(res3)
        self.assertEqual(res3.command, "git restore --staged main.py")

    def test_commit_queries(self):
        res1 = match_heuristic("committa dicendo 'fix auth issue'", self.ctx)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.command, 'git commit -m "fix auth issue"')

        res2 = match_heuristic("committa tutto con messaggio 'aggiornamento generale'", self.ctx)
        self.assertIsNotNone(res2)
        self.assertEqual(res2.command, 'git add -A && git commit -m "aggiornamento generale"')

        res3 = match_heuristic("modifica l'ultimo commit", self.ctx)
        self.assertIsNotNone(res3)
        self.assertEqual(res3.command, "git commit --amend")

    def test_descriptive_commit_queries(self):
        res1 = match_heuristic("descrivi un commit con tutte le informazioni dell'utente e dei files", self.ctx)
        self.assertIsNotNone(res1)
        self.assertIn("git", res1.command)
        self.assertIn("commit", res1.command)
        self.assertIn("Autore:", res1.preview)
        self.assertIn("Data:", res1.preview)

        res2 = match_heuristic("committa con tutte le informazioni dei file e dell'utente", self.ctx)
        self.assertIsNotNone(res2)
        self.assertIn("commit", res2.command)

        # Test metadata extraction directly
        user_info = self.ctx.get_user_info()
        self.assertIn("name", user_info)
        self.assertIn("email", user_info)
        self.assertIn("hostname", user_info)

    def test_branch_queries(self):
        res1 = match_heuristic("crea un nuovo branch chiamato feature/login", self.ctx)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.command, "git branch feature/login")

        res2 = match_heuristic("crea e spostati sul branch dev", self.ctx)
        self.assertIsNotNone(res2)
        self.assertEqual(res2.command, "git checkout -b dev")

        res3 = match_heuristic("spostati sul branch main", self.ctx)
        self.assertIsNotNone(res3)
        self.assertEqual(res3.command, "git checkout main")

        res4 = match_heuristic("elenca tutti i branch", self.ctx)
        self.assertIsNotNone(res4)
        self.assertEqual(res4.command, "git branch -a")

        res5 = match_heuristic("elimina il branch feature/old", self.ctx)
        self.assertIsNotNone(res5)
        self.assertEqual(res5.command, "git branch -d feature/old")
        self.assertTrue(res5.is_destructive)

        res6 = match_heuristic("elimina forzatamente il branch feature/corrupted", self.ctx)
        self.assertIsNotNone(res6)
        self.assertEqual(res6.command, "git branch -D feature/corrupted")
        self.assertTrue(res6.is_destructive)

    def test_sync_queries(self):
        res1 = match_heuristic("invia le modifiche al server", self.ctx)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.command, "git push origin feature/test")

        res2 = match_heuristic("scarica le ultime modifiche dal server", self.ctx)
        self.assertIsNotNone(res2)
        self.assertEqual(res2.command, "git pull")

        res3 = match_heuristic("controlla novità senza applicarle", self.ctx)
        self.assertIsNotNone(res3)
        self.assertEqual(res3.command, "git fetch --all")

        res4 = match_heuristic("mostra i server remoti", self.ctx)
        self.assertIsNotNone(res4)
        self.assertEqual(res4.command, "git remote -v")

    def test_stash_queries(self):
        res1 = match_heuristic("metti da parte le modifiche temporaneamente", self.ctx)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.command, "git stash")

        res2 = match_heuristic("recupera le modifiche messe da parte", self.ctx)
        self.assertIsNotNone(res2)
        self.assertEqual(res2.command, "git stash pop")

        res3 = match_heuristic("elenco stash", self.ctx)
        self.assertIsNotNone(res3)
        self.assertEqual(res3.command, "git stash list")

        res4 = match_heuristic("cancella stash", self.ctx)
        self.assertIsNotNone(res4)
        self.assertEqual(res4.command, "git stash drop")
        self.assertTrue(res4.is_destructive)

    def test_undo_destructive_queries(self):
        res1 = match_heuristic("annulla tutte le modifiche non salvate", self.ctx)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.command, "git restore . && git clean -fd")
        self.assertTrue(res1.is_destructive)

        res2 = match_heuristic("annulla l'ultimo commit tenendo le modifiche", self.ctx)
        self.assertIsNotNone(res2)
        self.assertEqual(res2.command, "git reset --soft HEAD~1")

        res3 = match_heuristic("cancella l'ultimo commit", self.ctx)
        self.assertIsNotNone(res3)
        self.assertEqual(res3.command, "git reset --hard HEAD~1")
        self.assertTrue(res3.is_destructive)

        res4 = match_heuristic("annulla modifiche al file config.json", self.ctx)
        self.assertIsNotNone(res4)
        self.assertEqual(res4.command, "git restore config.json")
        self.assertTrue(res4.is_destructive)

    def test_merge_rebase(self):
        res1 = match_heuristic("unisci il branch dev", self.ctx)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.command, "git merge dev")

        res2 = match_heuristic("abortisci il merge", self.ctx)
        self.assertIsNotNone(res2)
        self.assertEqual(res2.command, "git merge --abort")

        res3 = match_heuristic("fai il rebase su main", self.ctx)
        self.assertIsNotNone(res3)
        self.assertEqual(res3.command, "git rebase main")

    def test_logs_and_history(self):
        res1 = match_heuristic("mostrami gli ultimi 5 commit", self.ctx)
        self.assertIsNotNone(res1)
        self.assertIn("git log -n 5", res1.command)

        res2 = match_heuristic("mostra i commit ad albero", self.ctx)
        self.assertIsNotNone(res2)
        self.assertIn("git log --graph --oneline", res2.command)

    def test_tags_init_clone(self):
        res1 = match_heuristic("crea tag v1.0.0", self.ctx)
        self.assertIsNotNone(res1)
        self.assertEqual(res1.command, "git tag v1.0.0")

        res2 = match_heuristic("elenca tag", self.ctx)
        self.assertIsNotNone(res2)
        self.assertEqual(res2.command, "git tag -l")

        res3 = match_heuristic("inizializza il repository git", self.ctx)
        self.assertIsNotNone(res3)
        self.assertEqual(res3.command, "git init")

        res4 = match_heuristic("clona https://github.com/user/repo.git", self.ctx)
        self.assertIsNotNone(res4)
        self.assertEqual(res4.command, "git clone https://github.com/user/repo.git")


if __name__ == "__main__":
    unittest.main()
