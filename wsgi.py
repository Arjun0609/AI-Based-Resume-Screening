from app import app, ResumeAnalysisSystem, init_server

system = ResumeAnalysisSystem()
init_server(system)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8500)