data("phone_offers")

politeness::politeness(phone_offers$message)

install.packages("spacyr")
PYTHON_PATH = "C:\\Users\\andre\\AppData\\Local\\Programs\\Python\\Python39\\python.exe"
'C:\\Users\\andre\\AppData\\Local\\Programs\\Python\\Python39'
spacyr::spacy_initialize(python_executable = "PYTHON_PATH")
politeness::politeness(phone_offers$message, parser="spacy")

politeness::politenessPlot(politeness::politeness(phone_offers$message),
                           split=phone_offers$condition,
                           split_levels = c("Warm","Tough"),
                           split_name = "Condition")