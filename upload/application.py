# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 03:21:54 2018

@author: Dheeraj
"""
#import mysql.connector
import random
from flask import Flask, render_template,request,jsonify

app = Flask(__name__)
sentence_to_test=''


def getlongestword(sentence):
	words=sentence.split(" ")
	lenthlist=[(len(w),w) for w in words]
	return sorted(lenthlist)[-1][1]


@app.route('/')
def menu():
 
 return  render_template('homepage.html')
    

#main menu
@app.route('/words',methods=['POST'])        
def selection_table():
 #words=[]
 #data=[]
 #model_data=[]   
 #cnx = mysql.connector.connect(user='root',password='',host='127.0.0.1', database='words')
 #cursor=cnx.cursor()
 #query=("SELECT `word` FROM `content` ORDER BY RAND() LIMIT 10")
 #cursor.execute(query) 
 
 words_candidates=("i got you need me how old are you".split())*5
 words_candidates=[
"a",
"act",
"ad",
"add",
"age",
"ago",
"ah",
"aid",
"aim",
"air",
"all",
"AM",
"and",
"any",
"arm",
"art",
"as",
"ask",
"at",
"bad",
"bag",
"ban",
"bar",
"be",
"bed",
"bet",
"big",
"bit",
"box",
"boy",
"bus",
"but",
"buy",
"by",
"can",
"cap",
"car",
"cat",
"CEO",
"cop",
"cow",
"cry",
"cup",
"cut",
"dad",
"day",
"die",
"dig",
"DNA",
"do",
"dog",
"dry",
"due",
"ear",
"eat",
"egg",
"end",
"era",
"etc",
"eye",
"fan",
"far",
"fat",
"fee",
"few",
"fit",
"fix",
"fly",
"for",
"fun",
"gap",
"gas",
"gay",
"get",
"go",
"God",
"gun",
"guy",
"hat",
"he",
"her",
"hey",
"hi",
"him",
"hip",
"his",
"hit",
"hot",
"how",
"I",
"ice",
"ie",
"if",
"ill",
"in",
"it",
"its",
"jet",
"Jew",
"job",
"joy",
"key",
"kid",
"lab",
"lap",
"law",
"lay",
"leg",
"let",
"lie",
"lip",
"lot",
"low",
"mad",
"man",
"map",
"may",
"me",
"mix",
"mom",
"Mr",
"Mrs",
"Ms",
"my",
"net",
"new",
"no",
"nod",
"nor",
"not",
"now",
"nut",
"odd",
"of",
"off",
"oh",
"oil",
"ok",
"old",
"on",
"one",
"or",
"our",
"out",
"owe",
"own",
"pan",
"pay",
"PC",
"per",
"pet",
"pie",
"pop",
"pot",
"put",
"raw",
"red",
"rid",
"row",
"rub",
"run",
"sad",
"say",
"sea",
"see",
"set",
"sex",
"she",
"sin",
"sir",
"sit",
"six",
"ski",
"sky",
"so",
"son",
"sue",
"sun",
"tap",
"tax",
"tea",
"ten",
"the",
"tie",
"tip",
"to",
"toe",
"too",
"top",
"toy",
"try",
"TV",
"two",
"up",
"us",
"use",
"via",
"vs",
"war",
"way",
"we",
"wet",
"who",
"why",
"win",
"yes",
"yet",
"you",],["abandon",
"ability",
"abortion",
"about",
"above",
"abroad",
"absence",
"absolute",
"absorb",
"abuse",
"academic",
"accept",
"access",
"accident",
"accompany",
"according",
"account",
"accurate",
"accuse",
"achieve",
"acquire",
"across",
"action",
"active",
"activist",
"activity",
"actor",
"actress",
"actual",
"actually",
"adapt",
"addition",
"address",
"adequate",
"adjust",
"admire",
"admission",
"admit",
"adopt",
"adult",
"advance",
"advanced",
"advantage",
"adventure",
"advice",
"advise",
"adviser",
"advocate",
"affair",
"affect",
"afford",
"afraid",
"African",
"after",
"afternoon",
"again",
"against",
"agency",
"agenda",
"agent",
"agree",
"agreement",
"ahead",
"aircraft",
"airline",
"airport",
"album",
"alcohol",
"alive",
"alliance",
"allow",
"almost",
"alone",
"along",
"already",
"alter",
"although",
"always",
"amazing",
"American",
"among",
"amount",
"analysis",
"analyst",
"analyze",
"ancient",
"anger",
"angle",
"angry",
"animal",
"announce",
"annual",
"another",
"answer",
"anxiety",
"anybody",
"anymore",
"anyone",
"anything",
"anyway",
"anywhere",
"apart",
"apartment",
"apparent",
"appeal",
"appear",
"apple",
"apply",
"appoint",
"approach",
"approval",
"approve",
"architect",
"argue",
"argument",
"arise",
"armed",
"around",
"arrange",
"arrest",
"arrival",
"arrive",
"article",
"artist",
"artistic",
"Asian",
"aside",
"asleep",
"aspect",
"assault",
"assert",
"assess",
"asset",
"assign",
"assist",
"assistant",
"associate",
"assume",
"assure",
"athlete",
"athletic",
"attach",
"attack",
"attempt",
"attend",
"attention",
"attitude",
"attorney",
"attract",
"attribute",
"audience",
"author",
"authority",
"available",
"average",
"avoid",
"award",
"aware",
"awareness",
"awful",
"badly",
"balance",
"barely",
"barrel",
"barrier",
"baseball",
"basic",
"basically",
"basis",
"basket",
"bathroom",
"battery",
"battle",
"beach",
"beautiful",
"beauty",
"because",
"become",
"bedroom",
"before",
"begin",
"beginning",
"behavior",
"behind",
"being",
"belief",
"believe",
"belong",
"below",
"bench",
"beneath",
"benefit",
"beside",
"besides",
"better",
"between",
"beyond",
"Bible",
"billion",
"birth",
"birthday",
"black",
"blade",
"blame",
"blanket",
"blind",
"block",
"blood",
"board",
"bombing",
"border",
"borrow",
"bother",
"bottle",
"bottom",
"boundary",
"boyfriend",
"brain",
"branch",
"brand",
"bread",
"break",
"breakfast",
"breast",
"breath",
"breathe",
"brick",
"bridge",
"brief",
"briefly",
"bright",
"brilliant",
"bring",
"British",
"broad",
"broken",
"brother",
"brown",
"brush",
"budget",
"build",
"building",
"bullet",
"bunch",
"burden",
"business",
"butter",
"button",
"buyer",
"cabin",
"cabinet",
"cable",
"calculate",
"camera",
"campaign",
"campus",
"Canadian",
"cancer",
"candidate",
"capable",
"capacity",
"capital",
"captain",
"capture",
"carbon",
"career",
"careful",
"carefully",
"carrier",
"carry",
"catch",
"category",
"Catholic",
"cause",
"ceiling",
"celebrate",
"celebrity",
"center",
"central",
"century",
"ceremony",
"certain",
"certainly",
"chain",
"chair",
"chairman",
"challenge",
"chamber",
"champion",
"chance",
"change",
"changing",
"channel",
"chapter",
"character",
"charge",
"charity",
"chart",
"chase",
"cheap",
"check",
"cheek",
"cheese",
"chemical",
"chest",
"chicken",
"chief",
"child",
"childhood",
"Chinese",
"chocolate",
"choice",
"choose",
"Christian",
"Christmas",
"church",
"cigarette",
"circle",
"citizen",
"civil",
"civilian",
"claim",
"class",
"classic",
"classroom",
"clean",
"clear",
"clearly",
"client",
"climate",
"climb",
"clinic",
"clinical",
"clock",
"close",
"closely",
"closer",
"clothes",
"clothing",
"cloud",
"cluster",
"coach",
"coalition",
"coast",
"coffee",
"cognitive",
"collapse",
"colleague",
"collect",
"college",
"colonial",
"color",
"column",
"combine",
"comedy",
"comfort",
"command",
"commander",
"comment",
"commit",
"committee",
"common",
"community",
"company",
"compare",
"compete",
"complain",
"complaint",
"complete",
"complex",
"component",
"compose",
"computer",
"concept",
"concern",
"concerned",
"concert",
"conclude",
"concrete",
"condition",
"conduct",
"confident",
"confirm",
"conflict",
"confront",
"confusion",
"Congress",
"connect",
"consensus",
"consider",
"consist",
"constant",
"construct",
"consume",
"consumer",
"contact",
"contain",
"container",
"content",
"contest",
"context",
"continue",
"continued",
"contract",
"contrast",
"control",
"convert",
"convince",
"cookie",
"cooking",
"corner",
"corporate",
"correct",
"cotton",
"couch",
"could",
"council",
"counselor",
"count",
"counter",
"country",
"county",
"couple",
"courage",
"course",
"court",
"cousin",
"cover",
"coverage",
"crack",
"craft",
"crash",
"crazy",
"cream",
"create",
"creation",
"creative",
"creature",
"credit",
"crime",
"criminal",
"crisis",
"criteria",
"critic",
"critical",
"criticism",
"criticize",
"cross",
"crowd",
"crucial",
"cultural",
"culture",
"curious",
"current",
"currently",
"custom",
"customer",
"cycle",
"daily",
"damage",
"dance",
"danger",
"dangerous",
"darkness",
"daughter",
"dealer",
"death",
"debate",
"decade",
"decide",
"decision",
"declare",
"decline",
"decrease",
"deeply",
"defeat",
"defend",
"defendant",
"defense",
"defensive",
"deficit",
"define",
"degree",
"delay",
"deliver",
"delivery",
"demand",
"democracy",
"Democrat",
"depend",
"dependent",
"depending",
"depict",
"depth",
"deputy",
"derive",
"describe",
"desert",
"deserve",
"design",
"designer",
"desire",
"desperate",
"despite",
"destroy",
"detail",
"detailed",
"detect",
"determine",
"develop",
"device",
"devote",
"dialogue",
"differ",
"different",
"difficult",
"digital",
"dimension",
"dining",
"dinner",
"direct",
"direction",
"directly",
"director",
"dirty",
"disagree",
"disappear",
"disaster",
"discourse",
"discover",
"discovery",
"discuss",
"disease",
"dismiss",
"disorder",
"display",
"dispute",
"distance",
"distant",
"distinct",
"district",
"diverse",
"diversity",
"divide",
"division",
"divorce",
"doctor",
"document",
"domestic",
"dominant",
"dominate",
"double",
"doubt",
"downtown",
"dozen",
"draft",
"drama",
"dramatic",
"drawing",
"dream",
"dress",
"drink",
"drive",
"driver",
"during",
"eager",
"early",
"earnings",
"earth",
"easily",
"eastern",
"economic",
"economics",
"economist",
"economy",
"edition",
"editor",
"educate",
"education",
"educator",
"effect",
"effective",
"efficient",
"effort",
"eight",
"either",
"elderly",
"elect",
"election",
"electric",
"element",
"eliminate",
"elite",
"elsewhere",
"e-mail",
"embrace",
"emerge",
"emergency",
"emission",
"emotion",
"emotional",
"emphasis",
"emphasize",
"employ",
"employee",
"employer",
"empty",
"enable",
"encounter",
"encourage",
"enemy",
"energy",
"engage",
"engine",
"engineer",
"English",
"enhance",
"enjoy",
"enormous",
"enough",
"ensure",
"enter",
"entire",
"entirely",
"entrance",
"entry",
"episode",
"equal",
"equally",
"equipment",
"error",
"escape",
"essay",
"essential",
"establish",
"estate",
"estimate",
"ethics",
"ethnic",
"European",
"evaluate",
"evening",
"event",
"every",
"everybody",
"everyday",
"everyone",
"evidence",
"evolution",
"evolve",
"exact",
"exactly",
"examine",
"example",
"exceed",
"excellent",
"except",
"exception",
"exchange",
"exciting",
"executive",
"exercise",
"exhibit",
"exist",
"existence",
"existing",
"expand",
"expansion",
"expect",
"expense",
"expensive",
"expert",
"explain",
"explode",
"explore",
"explosion",
"expose",
"exposure",
"express",
"extend",
"extension",
"extensive",
"extent",
"external",
"extra",
"extreme",
"extremely",
"fabric",
"facility",
"factor",
"factory",
"faculty",
"failure",
"fairly",
"faith",
"false",
"familiar",
"family",
"famous",
"fantasy",
"farmer",
"fashion",
"father",
"fault",
"favor",
"favorite",
"feature",
"federal",
"feeling",
"fellow",
"female",
"fence",
"fewer",
"fiber",
"fiction",
"field",
"fifteen",
"fifth",
"fifty",
"fight",
"fighter",
"fighting",
"figure",
"final",
"finally",
"finance",
"financial",
"finding",
"finger",
"finish",
"first",
"fishing",
"fitness",
"flame",
"flavor",
"flesh",
"flight",
"float",
"floor",
"flower",
"focus",
"follow",
"following",
"football",
"force",
"foreign",
"forest",
"forever",
"forget",
"formal",
"formation",
"former",
"formula",
"forth",
"fortune",
"forward",
"found",
"founder",
"fourth",
"frame",
"framework",
"freedom",
"freeze",
"French",
"frequency",
"frequent",
"fresh",
"friend",
"friendly",
"front",
"fruit",
"fully",
"function",
"funding",
"funeral",
"funny",
"furniture",
"future",
"galaxy",
"gallery",
"garage",
"garden",
"garlic",
"gather",
"gender",
"general",
"generally",
"generate",
"genetic",
"gentleman",
"gently",
"German",
"gesture",
"ghost",
"giant",
"gifted",
"given",
"glance",
"glass",
"global",
"glove",
"golden",
"governor",
"grade",
"gradually",
"graduate",
"grain",
"grand",
"grant",
"grass",
"grave",
"great",
"greatest",
"green",
"grocery",
"ground",
"group",
"growing",
"growth",
"guarantee",
"guard",
"guess",
"guest",
"guide",
"guideline",
"guilty",
"habit",
"habitat",
"handful",
"handle",
"happen",
"happy",
"hardly",
"headline",
"health",
"healthy",
"hearing",
"heart",
"heaven",
"heavily",
"heavy",
"height",
"hello",
"helpful",
"heritage",
"herself",
"highlight",
"highly",
"highway",
"himself",
"historian",
"historic",
"history",
"holiday",
"homeless",
"honest",
"honey",
"honor",
"horizon",
"horror",
"horse",
"hospital",
"hotel",
"house",
"household",
"housing",
"however",
"human",
"humor",
"hundred",
"hungry",
"hunter",
"hunting",
"husband",
"ideal",
"identify",
"identity",
"ignore",
"illegal",
"illness",
"image",
"imagine",
"immediate",
"immigrant",
"impact",
"implement",
"imply",
"important",
"impose",
"impress",
"improve",
"incentive",
"incident",
"include",
"including",
"income",
"increase",
"increased",
"indeed",
"index",
"Indian",
"indicate",
"industry",
"infant",
"infection",
"inflation",
"influence",
"inform",
"initial",
"initially",
"injury",
"inner",
"innocent",
"inquiry",
"inside",
"insight",
"insist",
"inspire",
"install",
"instance",
"instead",
"insurance",
"intend",
"intense",
"intensity",
"intention",
"interest",
"internal",
"Internet",
"interpret",
"interview",
"introduce",
"invasion",
"invest",
"investor",
"invite",
"involve",
"involved",
"Iraqi",
"Irish",
"Islamic",
"island",
"Israeli",
"issue",
"Italian",
"itself",
"jacket",
"Japanese",
"Jewish",
"joint",
"journal",
"journey",
"judge",
"judgment",
"juice",
"junior",
"justice",
"justify",
"killer",
"killing",
"kitchen",
"knife",
"knock",
"knowledge",
"label",
"labor",
"landscape",
"language",
"large",
"largely",
"later",
"Latin",
"latter",
"laugh",
"launch",
"lawsuit",
"lawyer",
"layer",
"leader",
"leading",
"league",
"learn",
"learning",
"least",
"leather",
"leave",
"legacy",
"legal",
"legend",
"lemon",
"length",
"lesson",
"letter",
"level",
"liberal",
"library",
"license",
"lifestyle",
"lifetime",
"light",
"likely",
"limit",
"limited",
"listen",
"literally",
"literary",
"little",
"living",
"local",
"locate",
"location",
"long-term",
"loose",
"lovely",
"lover",
"lower",
"lucky",
"lunch",
"machine",
"magazine",
"mainly",
"maintain",
"major",
"majority",
"maker",
"makeup",
"manage",
"manager",
"manner",
"margin",
"market",
"marketing",
"marriage",
"married",
"marry",
"massive",
"master",
"match",
"material",
"matter",
"maybe",
"mayor",
"meaning",
"meanwhile",
"measure",
"mechanism",
"media",
"medical",
"medicine",
"medium",
"meeting",
"member",
"memory",
"mental",
"mention",
"merely",
"message",
"metal",
"meter",
"method",
"Mexican",
"middle",
"might",
"military",
"million",
"minister",
"minor",
"minority",
"minute",
"miracle",
"mirror",
"missile",
"mission",
"mistake",
"mixture",
"mm-hmm",
"model",
"moderate",
"modern",
"modest",
"moment",
"money",
"monitor",
"month",
"moral",
"moreover",
"morning",
"mortgage",
"mostly",
"mother",
"motion",
"motor",
"mount",
"mountain",
"mouse",
"mouth",
"movement",
"movie",
"multiple",
"murder",
"muscle",
"museum",
"music",
"musical",
"musician",
"Muslim",
"mutual",
"myself",
"mystery",
"naked",
"narrative",
"narrow",
"nation",
"national",
"native",
"natural",
"naturally",
"nature",
"nearby",
"nearly",
"necessary",
"negative",
"negotiate",
"neighbor",
"neither",
"nerve",
"nervous",
"network",
"never",
"newly",
"newspaper",
"night",
"nobody",
"noise",
"normal",
"normally",
"north",
"northern",
"nothing",
"notice",
"notion",
"novel",
"nowhere",
"nuclear",
"number",
"numerous",
"nurse",
"object",
"objective",
"observe",
"observer",
"obtain",
"obvious",
"obviously",
"occasion",
"occupy",
"occur",
"ocean",
"offense",
"offensive",
"offer",
"office",
"officer",
"official",
"often",
"Olympic",
"ongoing",
"onion",
"online",
"opening",
"operate",
"operating",
"operation",
"operator",
"opinion",
"opponent",
"oppose",
"opposite",
"option",
"orange",
"order",
"ordinary",
"organic",
"organize",
"origin",
"original",
"other",
"others",
"otherwise",
"ought",
"ourselves",
"outcome",
"outside",
"overall",
"overcome",
"overlook",
"owner",
"package",
"painful",
"paint",
"painter",
"painting",
"panel",
"paper",
"parent",
"parking",
"partly",
"partner",
"party",
"passage",
"passenger",
"passion",
"patch",
"patient",
"pattern",
"pause",
"payment",
"peace",
"penalty",
"people",
"pepper",
"perceive",
"perfect",
"perfectly",
"perform",
"perhaps",
"period",
"permanent",
"permit",
"person",
"personal",
"personnel",
"persuade",
"phase",
"phone",
"photo",
"phrase",
"physical",
"physician",
"piano",
"picture",
"piece",
"pilot",
"pitch",
"place",
"plane",
"planet",
"planning",
"plant",
"plastic",
"plate",
"platform",
"player",
"please",
"pleasure",
"plenty",
"pocket",
"poetry",
"point",
"police",
"policy",
"political",
"politics",
"pollution",
"popular",
"porch",
"portion",
"portrait",
"portray",
"position",
"positive",
"possess",
"possible",
"possibly",
"potato",
"potential",
"pound",
"poverty",
"powder",
"power",
"powerful",
"practical",
"practice",
"prayer",
"precisely",
"predict",
"prefer",
"pregnancy",
"pregnant",
"prepare",
"presence",
"present",
"preserve",
"president",
"press",
"pressure",
"pretend",
"pretty",
"prevent",
"previous",
"price",
"pride",
"priest",
"primarily",
"primary",
"prime",
"principal",
"principle",
"print",
"prior",
"priority",
"prison",
"prisoner",
"privacy",
"private",
"probably",
"problem",
"procedure",
"proceed",
"process",
"produce",
"producer",
"product",
"professor",
"profile",
"profit",
"program",
"progress",
"project",
"prominent",
"promise",
"promote",
"prompt",
"proof",
"proper",
"properly",
"property",
"proposal",
"propose",
"proposed",
"prospect",
"protect",
"protein",
"protest",
"proud",
"prove",
"provide",
"provider",
"province",
"provision",
"public",
"publicly",
"publish",
"publisher",
"purchase",
"purpose",
"pursue",
"qualify",
"quality",
"quarter",
"question",
"quick",
"quickly",
"quiet",
"quietly",
"quite",
"quote",
"racial",
"radical",
"radio",
"raise",
"range",
"rapid",
"rapidly",
"rarely",
"rather",
"rating",
"ratio",
"reach",
"react",
"reaction",
"reader",
"reading",
"ready",
"reality",
"realize",
"really",
"reason",
"recall",
"receive",
"recent",
"recently",
"recipe",
"recognize",
"recommend",
"record",
"recording",
"recover",
"recovery",
"recruit",
"reduce",
"reduction",
"refer",
"reference",
"reflect",
"reform",
"refugee",
"refuse",
"regard",
"regarding",
"regime",
"region",
"regional",
"register",
"regular",
"regularly",
"regulate",
"reinforce",
"reject",
"relate",
"relation",
"relative",
"relax",
"release",
"relevant",
"relief",
"religion",
"religious",
"remain",
"remaining",
"remember",
"remind",
"remote",
"remove",
"repeat",
"replace",
"reply",
"report",
"reporter",
"represent",
"request",
"require",
"research",
"resemble",
"resident",
"resist",
"resolve",
"resort",
"resource",
"respect",
"respond",
"response",
"restore",
"result",
"retain",
"retire",
"return",
"reveal",
"revenue",
"review",
"rhythm",
"rifle",
"right",
"river",
"romantic",
"rough",
"roughly",
"round",
"route",
"routine",
"running",
"rural",
"Russian",
"sacred",
"safety",
"salad",
"salary",
"sales",
"sample",
"sanction",
"satellite",
"satisfy",
"sauce",
"saving",
"scale",
"scandal",
"scared",
"scenario",
"scene",
"schedule",
"scheme",
"scholar",
"school",
"science",
"scientist",
"scope",
"score",
"scream",
"screen",
"script",
"search",
"season",
"second",
"secret",
"secretary",
"section",
"sector",
"secure",
"security",
"segment",
"seize",
"select",
"selection",
"Senate",
"senator",
"senior",
"sense",
"sensitive",
"sentence",
"separate",
"sequence",
"series",
"serious",
"seriously",
"serve",
"service",
"session",
"setting",
"settle",
"seven",
"several",
"severe",
"sexual",
"shade",
"shadow",
"shake",
"shall",
"shape",
"share",
"sharp",
"sheet",
"shelf",
"shell",
"shelter",
"shift",
"shine",
"shirt",
"shock",
"shoot",
"shooting",
"shopping",
"shore",
"short",
"shortly",
"should",
"shoulder",
"shout",
"shower",
"shrug",
"sight",
"signal",
"silence",
"silent",
"silver",
"similar",
"similarly",
"simple",
"simply",
"since",
"singer",
"single",
"sister",
"situation",
"skill",
"slave",
"sleep",
"slice",
"slide",
"slight",
"slightly",
"slowly",
"small",
"smart",
"smell",
"smile",
"smoke",
"smooth",
"so-called",
"soccer",
"social",
"society",
"software",
"solar",
"soldier",
"solid",
"solution",
"solve",
"somebody",
"somehow",
"someone",
"something",
"sometimes",
"somewhat",
"somewhere",
"sorry",
"sound",
"source",
"south",
"southern",
"Soviet",
"space",
"Spanish",
"speak",
"speaker",
"special",
"species",
"specific",
"speech",
"speed",
"spend",
"spending",
"spirit",
"spiritual",
"split",
"spokesman",
"sport",
"spread",
"spring",
"square",
"squeeze",
"stability",
"stable",
"staff",
"stage",
"stair",
"stake",
"stand",
"standard",
"standing",
"stare",
"start",
"state",
"statement",
"station",
"status",
"steady",
"steal",
"steel",
"stick",
"still",
"stock",
"stomach",
"stone",
"storage",
"store",
"storm",
"story",
"straight",
"strange",
"stranger",
"strategic",
"strategy",
"stream",
"street",
"strength",
"stress",
"stretch",
"strike",
"string",
"strip",
"stroke",
"strong",
"strongly",
"structure",
"struggle",
"student",
"studio",
"study",
"stuff",
"stupid",
"style",
"subject",
"submit",
"substance",
"succeed",
"success",
"sudden",
"suddenly",
"suffer",
"sugar",
"suggest",
"suicide",
"summer",
"summit",
"super",
"supply",
"support",
"supporter",
"suppose",
"supposed",
"Supreme",
"surely",
"surface",
"surgery",
"surprise",
"surprised",
"surround",
"survey",
"survival",
"survive",
"survivor",
"suspect",
"sustain",
"swear",
"sweep",
"sweet",
"swing",
"switch",
"symbol",
"symptom",
"system",
"table",
"tactic",
"talent",
"target",
"taste",
"taxpayer",
"teach",
"teacher",
"teaching",
"teaspoon",
"technical",
"technique",
"teenager",
"telephone",
"telescope",
"temporary",
"tendency",
"tennis",
"tension",
"terms",
"terrible",
"territory",
"terror",
"terrorism",
"terrorist",
"testify",
"testimony",
"testing",
"thank",
"thanks",
"theater",
"their",
"theme",
"theory",
"therapy",
"there",
"therefore",
"these",
"thick",
"thing",
"think",
"thinking",
"third",
"thirty",
"those",
"though",
"thought",
"thousand",
"threat",
"threaten",
"three",
"throat",
"through",
"throw",
"ticket",
"tight",
"tired",
"tissue",
"title",
"tobacco",
"today",
"together",
"tomato",
"tomorrow",
"tongue",
"tonight",
"tooth",
"topic",
"total",
"totally",
"touch",
"tough",
"tourist",
"toward",
"towards",
"tower",
"trace",
"track",
"trade",
"tradition",
"traffic",
"tragedy",
"trail",
"train",
"training",
"transfer",
"transform",
"translate",
"travel",
"treat",
"treatment",
"treaty",
"trend",
"trial",
"tribe",
"trick",
"troop",
"trouble",
"truck",
"truly",
"trust",
"truth",
"tunnel",
"twelve",
"twenty",
"twice",
"typical",
"typically",
"ultimate",
"unable",
"uncle",
"under",
"undergo",
"uniform",
"union",
"unique",
"United",
"universal",
"universe",
"unknown",
"unless",
"unlike",
"unlikely",
"until",
"unusual",
"upper",
"urban",
"useful",
"usual",
"usually",
"utility",
"vacation",
"valley",
"valuable",
"value",
"variable",
"variation",
"variety",
"various",
"vegetable",
"vehicle",
"venture",
"version",
"versus",
"vessel",
"veteran",
"victim",
"victory",
"video",
"viewer",
"village",
"violate",
"violation",
"violence",
"violent",
"virtually",
"virtue",
"virus",
"visible",
"vision",
"visit",
"visitor",
"visual",
"vital",
"voice",
"volume",
"volunteer",
"voter",
"wander",
"warning",
"waste",
"watch",
"water",
"wealth",
"wealthy",
"weapon",
"weather",
"wedding",
"weekend",
"weekly",
"weigh",
"weight",
"welcome",
"welfare",
"western",
"whatever",
"wheel",
"whenever",
"where",
"whereas",
"whether",
"which",
"while",
"whisper",
"white",
"whole",
"whose",
"widely",
"willing",
"window",
"winner",
"winter",
"wisdom",
"withdraw",
"within",
"without",
"witness",
"woman",
"wonder",
"wonderful",
"wooden",
"worker",
"working",
"works",
"workshop",
"world",
"worried",
"worry",
"worth",
"would",
"wound",
"write",
"writer",
"writing",
"wrong",
"yellow",
"yesterday",
"yield",
"young",
"yours",
"yourself",
"youth"],["absolutely",
"accomplish",
"achievement",
"acknowledge",
"additional",
"adjustment",
"administration",
"administrator",
"adolescent",
"advertising",
"African-American",
"aggressive",
"agricultural",
"alternative",
"anniversary",
"anticipate",
"apparently",
"appearance",
"application",
"appointment",
"appreciate",
"appropriate",
"approximately",
"arrangement",
"assessment",
"assignment",
"assistance",
"association",
"assumption",
"atmosphere",
"attractive",
"background",
"basketball",
"biological",
"capability",
"celebration",
"championship",
"characteristic",
"characterize",
"cholesterol",
"circumstance",
"collection",
"collective",
"combination",
"comfortable",
"commercial",
"commission",
"commitment",
"communicate",
"communication",
"comparison",
"competition",
"competitive",
"competitor",
"completely",
"complicated",
"composition",
"comprehensive",
"concentrate",
"concentration",
"conclusion",
"conference",
"confidence",
"congressional",
"connection",
"consciousness",
"consequence",
"conservative",
"considerable",
"consideration",
"consistent",
"constantly",
"constitute",
"constitutional",
"construction",
"consultant",
"consumption",
"contemporary",
"contribute",
"contribution",
"controversial",
"controversy",
"convention",
"conventional",
"conversation",
"conviction",
"cooperation",
"corporation",
"correspondent",
"curriculum",
"definitely",
"definition",
"democratic",
"demonstrate",
"demonstration",
"department",
"depression",
"description",
"destruction",
"developing",
"development",
"difference",
"differently",
"difficulty",
"disability",
"discipline",
"discrimination",
"discussion",
"distinction",
"distinguish",
"distribute",
"distribution",
"dramatically",
"educational",
"effectively",
"efficiency",
"electricity",
"electronic",
"elementary",
"employment",
"enforcement",
"engineering",
"enterprise",
"entertainment",
"environment",
"environmental",
"especially",
"essentially",
"establishment",
"evaluation",
"eventually",
"everything",
"everywhere",
"examination",
"exhibition",
"expectation",
"experience",
"experiment",
"explanation",
"expression",
"extraordinary",
"foundation",
"frequently",
"friendship",
"frustration",
"fundamental",
"furthermore",
"generation",
"girlfriend",
"government",
"grandfather",
"grandmother",
"headquarters",
"helicopter",
"historical",
"hypothesis",
"identification",
"illustrate",
"imagination",
"immediately",
"immigration",
"implication",
"importance",
"impossible",
"impression",
"impressive",
"improvement",
"incorporate",
"increasing",
"increasingly",
"incredible",
"independence",
"independent",
"indication",
"individual",
"industrial",
"information",
"ingredient",
"initiative",
"institution",
"institutional",
"instruction",
"instructor",
"instrument",
"intellectual",
"intelligence",
"interaction",
"interested",
"interesting",
"international",
"interpretation",
"intervention",
"introduction",
"investigate",
"investigation",
"investigator",
"investment",
"involvement",
"journalist",
"laboratory",
"leadership",
"legislation",
"legitimate",
"limitation",
"literature",
"maintenance",
"management",
"manufacturer",
"manufacturing",
"measurement",
"medication",
"membership",
"motivation",
"necessarily",
"negotiation",
"neighborhood",
"nevertheless",
"nomination",
"nonetheless",
"obligation",
"observation",
"occasionally",
"occupation",
"opportunity",
"opposition",
"organization",
"orientation",
"originally",
"Palestinian",
"participant",
"participate",
"participation",
"particular",
"particularly",
"partnership",
"percentage",
"perception",
"performance",
"permission",
"personality",
"personally",
"perspective",
"phenomenon",
"philosophy",
"photograph",
"photographer",
"physically",
"politically",
"politician",
"population",
"possibility",
"potentially",
"preference",
"preparation",
"prescription",
"presentation",
"presidential",
"previously",
"production",
"profession",
"professional",
"proportion",
"prosecutor",
"protection",
"psychological",
"psychologist",
"psychology",
"publication",
"punishment",
"quarterback",
"reasonable",
"recognition",
"recommendation",
"reflection",
"regardless",
"regulation",
"relationship",
"relatively",
"remarkable",
"repeatedly",
"representation",
"representative",
"Republican",
"reputation",
"requirement",
"researcher",
"reservation",
"resistance",
"resolution",
"respondent",
"responsibility",
"responsible",
"restaurant",
"restriction",
"retirement",
"revolution",
"satisfaction",
"scholarship",
"scientific",
"settlement",
"significance",
"significant",
"significantly",
"sophisticated",
"specialist",
"specifically",
"statistics",
"strengthen",
"subsequent",
"substantial",
"successful",
"successfully",
"sufficient",
"suggestion",
"surprising",
"surprisingly",
"tablespoon",
"technology",
"television",
"temperature",
"themselves",
"throughout",
"tournament",
"traditional",
"transformation",
"transition",
"transportation",
"tremendous",
"ultimately",
"understand",
"understanding",
"unfortunately",
"university",
"vulnerable",
"widespread"]
 model_data=random.sample(words_candidates[1], 10)
 '''
 for word in cursor:
     
     data.append( str( word[0]))
     
     
     words.append(word)
     
     
 for(item)   in data:
     
     item1=item.replace('\r\n',"")
     item2=item1.replace("'","" )
     model_data.append(item2)
 
 print(model_data)
 '''
 #cursor.close()
 #cnx.close()

     

 
 return render_template('getwords.html',words=model_data)

@app.route('/learning',methods=['GET'])
def learning():
 #cnx=mysql.connector.connect(user='root',password='',host='127.0.0.1',database='words')
# cursor=cnx.cursor()
    
     
    
 a = request.args.get('a')#prounciation

 a = getlongestword(a)
    
 b=request.args.get('b')#actual
 b=b.strip()

 
 
 
 
 #cursor.execute("UPDATE content SET number_of_attempts=number_of_attempts+1 WHERE word='%s'" %(b))
 #cnx.commit()
 

    
 #if(a==b):
     
    # cursor.execute("UPDATE content SET correct_responses=correct_responses+1 WHERE word='%s'" %(b))
     #cnx.commit()
     
        
    
 #cnx.close()
 return jsonify(a=a,b=b)
        

@app.route('/phrases',methods=['POST'])

def phrases():
 #words=[]
 #data=[]
 #model_data=[]   
 #cnx = mysql.connector.connect(user='root',password='',host='127.0.0.1', database='phrases')
 #cursor=cnx.cursor()
 #query=("SELECT `sentence` FROM `phrase` ORDER BY RAND() LIMIT 10")
 #cursor.execute(query)
 words_candidates=("i got you need me how old are you".split())*5
 model_data=random.sample(words_candidates, 10)
 '''
 for word in cursor:
     
     data.append( str( word[0]))
     
     
     words.append(word)
     
     
 for(item)   in data:
     
     item1=item.replace('\r\n',"")
     item2=item1.replace("'","" )
     model_data.append(item2)
 
 print(model_data)
 
 cursor.close()
 cnx.close()
 '''

 return render_template('getphrases.html',words=model_data)

    


    


       
    

#once the selection_button is pressed:
#@app.route('/', methods=['POST'])
#def selection_result():
# sentence_to_test=request.selection['selection_table']
# return render_template('play_and_record.html',sentence=sentence_to_test)

#once the stop_test button is pressed:
#@app.route('/playrecord', methods=['POST'])
#def stoprecord():
# voice = request.form['raw_voice']
# sentence_google=request.form['google_translation']
# sentence_from_ml=ml_model(voice)
# score_,sentence_to_test=result_cal(sentence,sentence_google,sentence_from_ml)
 #return render_template('show_result.html', score=score_)

#once the test_remaining button is pressed:
#@app.route('/testremain', methods=['POST'])
#def testremain():
#return render_template('play_and_record.html',sentence=sentence_to_test)



if __name__ == '__main__':
 app.run(debug=True)