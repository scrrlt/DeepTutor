with open(r'D:\dev-hub\forks\DeepTutor\deeptutor\api\main.py', 'r', encoding='utf-8') as f:
    orig = f.read()

imported = 'from deeptutor.api.routers import ('
new_imp = imported + '\n    write,'
orig = orig.replace(imported, new_imp)

old_router = 'app.include_router(quiz_judge.router, prefix=\"/api/v1\", tags=[\"quiz-judge\"])'
new_router = old_router + '\napp.include_router(write.router, prefix=\"/api/v1/write\", tags=[\"write\"], dependencies=_auth)'
orig = orig.replace(old_router, new_router)

with open(r'D:\dev-hub\forks\DeepTutor\deeptutor\api\main.py', 'w', encoding='utf-8') as f:
    f.write(orig)
