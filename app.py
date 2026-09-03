# استيراد المكتبات المطلوبة
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd


# إنشاء تطبيق Flask
app = Flask(__name__)


# تحميل نموذج التعلم الآلي
model = joblib.load("heart_disease_model.pkl")


# تحديد المتغيرات التي يحتاجها النموذج
REQUIRED_FIELDS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# تحديد القيم الفئوية المسموح بها
ALLOWED_VALUES = {
    "sex": [
        "Male",
        "Female"
    ],

    "cp": [
        "typical angina",
        "atypical angina",
        "non-anginal",
        "asymptomatic"
    ],

    "fbs": [
        True,
        False
    ],

    "restecg": [
        "normal",
        "lv hypertrophy",
        "st-t abnormality"
    ],

    "exang": [
        True,
        False
    ],

    "slope": [
        "upsloping",
        "flat",
        "downsloping"
    ],

    "thal": [
        "normal",
        "fixed defect",
        "reversable defect"
    ]
}


# تحديد النطاقات المقبولة للمتغيرات الرقمية
NUMERIC_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "thalch": (50, 250),
    "oldpeak": (-10, 20),
    "ca": (0, 3)
}


# تحديد مستوى الخطورة اعتمادًا على احتمالية النموذج
def get_risk_level(probability):

    if probability < 0.30:

        return "منخفض"

    elif probability < 0.70:

        return "متوسط"

    else:

        return "مرتفع"


# استخراج أهم النتائج من بيانات المريض
def generate_patient_findings(data):

    findings = []


    # تحليل العمر
    if data["age"] >= 65:

        findings.append(
            "العمر المدخل ضمن فئة عمرية متقدمة ويُؤخذ ضمن التقييم العام لعوامل الخطورة."
        )

    elif data["age"] >= 55:

        findings.append(
            "العمر المدخل يمثل عاملًا ينبغي أخذه في الاعتبار عند تقييم خطر أمراض القلب."
        )


    # تحليل نوع ألم الصدر
    if data["cp"] == "typical angina":

        findings.append(
            "ألم الصدر المدخل مصنف كذبحة نموذجية."
        )

    elif data["cp"] == "atypical angina":

        findings.append(
            "ألم الصدر المدخل مصنف كذبحة غير نموذجية."
        )

    elif data["cp"] == "non-anginal":

        findings.append(
            "ألم الصدر المدخل مصنف كألم غير ذبحي."
        )

    elif data["cp"] == "asymptomatic":

        findings.append(
            "لا توجد أعراض ذبحة صدرية مذكورة ضمن هذا المتغير."
        )


    # تحليل ضغط الدم
    if data["trestbps"] >= 180:

        findings.append(
            "قيمة ضغط الدم أثناء الراحة المدخلة مرتفعة بشكل ملحوظ."
        )

    elif data["trestbps"] >= 140:

        findings.append(
            "قيمة ضغط الدم أثناء الراحة المدخلة مرتفعة."
        )

    elif data["trestbps"] >= 130:

        findings.append(
            "قيمة ضغط الدم أثناء الراحة المدخلة أعلى من المستوى المثالي."
        )


    # تحليل الكوليسترول
    if data["chol"] >= 240:

        findings.append(
            "مستوى الكوليسترول المدخل مرتفع."
        )

    elif data["chol"] >= 200:

        findings.append(
            "مستوى الكوليسترول المدخل أعلى من المستوى المرغوب."
        )


    # تحليل سكر الدم الصائم
    if data["fbs"] is True:

        findings.append(
            "قيمة سكر الدم الصائم المدخلة ضمن التصنيف المرتفع في البيانات."
        )


    # تحليل تخطيط القلب
    if data["restecg"] == "lv hypertrophy":

        findings.append(
            "نتيجة تخطيط القلب المدخلة تشير إلى تضخم البطين الأيسر."
        )

    elif data["restecg"] == "st-t abnormality":

        findings.append(
            "توجد تغيرات في ST-T ضمن نتيجة تخطيط القلب المدخلة."
        )


    # تحليل الذبحة المرتبطة بالجهد
    if data["exang"] is True:

        findings.append(
            "تم تسجيل وجود ذبحة مرتبطة بالجهد."
        )


    # تحليل قيمة Oldpeak
    if data["oldpeak"] >= 2:

        findings.append(
            "قيمة Oldpeak المدخلة مرتفعة نسبيًا."
        )

    elif data["oldpeak"] > 0:

        findings.append(
            "توجد قيمة Oldpeak موجبة ضمن بيانات الحالة."
        )


    # تحليل ميل ST
    if data["slope"] == "flat":

        findings.append(
            "ميل ST المدخل مصنف كمسطح."
        )

    elif data["slope"] == "downsloping":

        findings.append(
            "ميل ST المدخل مصنف كمنحدر إلى الأسفل."
        )


    # تحليل عدد الأوعية الرئيسية
    if data["ca"] > 0:

        findings.append(
            f"تم تسجيل {data['ca']} من الأوعية الرئيسية ضمن متغير ca."
        )


    # تحليل نتيجة Thal
    if data["thal"] == "fixed defect":

        findings.append(
            "نتيجة Thal المدخلة مصنفة كعيب ثابت."
        )

    elif data["thal"] == "reversable defect":

        findings.append(
            "نتيجة Thal المدخلة مصنفة كعيب قابل للعكس."
        )


    # إعادة النتائج
    return findings


# إنشاء تنبيهات مخصصة للطبيب
def generate_doctor_alerts(data, probability, risk_level):

    alerts = []


    # تحديد مستوى أولوية نتيجة النموذج
    if risk_level == "مرتفع":

        alerts.append({
            "priority": 1,
            "message": (
                f"نتيجة النموذج تشير إلى احتمال مرتفع ({probability * 100:.1f}%)؛ "
                "ينبغي إعطاء الحالة أولوية للتقييم السريري وربط النتيجة بالأعراض والفحوصات."
            )
        })

    elif risk_level == "متوسط":

        alerts.append({
            "priority": 2,
            "message": (
                f"نتيجة النموذج تشير إلى احتمال متوسط ({probability * 100:.1f}%)؛ "
                "ينبغي تفسير النتيجة مع الأعراض وعوامل الخطورة والنتائج السريرية."
            )
        })

    else:

        alerts.append({
            "priority": 3,
            "message": (
                f"نتيجة النموذج تشير إلى احتمال منخفض ({probability * 100:.1f}%)؛ "
                "لا ينبغي استخدام النتيجة وحدها لاستبعاد وجود مشكلة قلبية عند وجود أعراض."
            )
        })


    # رفع أولوية التنبيه عند وجود ألم صدر نموذجي
    if data["cp"] == "typical angina":

        alerts.append({
            "priority": 1,
            "message": (
                "ألم الصدر مصنف كذبحة نموذجية؛ "
                "ينبغي تقييم طبيعة الأعراض وارتباطها بالمجهود والأعراض المصاحبة."
            )
        })


    # التنبيه عند وجود ألم صدر غير نموذجي
    elif data["cp"] == "atypical angina":

        alerts.append({
            "priority": 2,
            "message": (
                "توجد أعراض ألم صدر غير نموذجية؛ "
                "ينبغي ربطها بالتاريخ المرضي وبقية عوامل الخطورة."
            )
        })


    # التنبيه عند وجود ذبحة مرتبطة بالجهد
    if data["exang"] is True:

        alerts.append({
            "priority": 1,
            "message": (
                "تم تسجيل أعراض مرتبطة بالجهد؛ "
                "ينبغي مراجعة علاقتها بالنشاط والأعراض القلبية المصاحبة."
            )
        })


    # جمع عوامل الخطورة القابلة للتعديل
    risk_factors = []


    if data["trestbps"] >= 140:

        risk_factors.append("ارتفاع ضغط الدم")


    if data["chol"] >= 200:

        risk_factors.append("ارتفاع الكوليسترول")


    if data["fbs"] is True:

        risk_factors.append("ارتفاع سكر الدم الصائم")


    # إنشاء تنبيه واحد بدل عدة تنبيهات متكررة
    if risk_factors:

        risk_text = "، ".join(
            risk_factors
        )

        alerts.append({
            "priority": 2,
            "message": (
                f"توجد عوامل خطورة قابلة للتعديل ({risk_text})؛ "
                "ينبغي تقييمها ضمن خطة الوقاية والمتابعة القلبية."
            )
        })


    # جمع نتائج اختبارات القلب غير الطبيعية
    cardiac_tests = []


    if data["restecg"] != "normal":

        cardiac_tests.append("تخطيط القلب")


    if data["oldpeak"] > 0:

        cardiac_tests.append("Oldpeak")


    if data["slope"] != "upsloping":

        cardiac_tests.append("ميل ST")


    if data["ca"] > 0:

        cardiac_tests.append("عدد الأوعية")


    if data["thal"] != "normal":

        cardiac_tests.append("Thal")


    # إنشاء تنبيه واحد للنتائج القلبية
    if cardiac_tests:

        tests_text = "، ".join(
            cardiac_tests
        )

        alerts.append({
            "priority": 2,
            "message": (
                f"توجد نتائج غير طبيعية أو غير اعتيادية في ({tests_text})؛ "
                "ينبغي تفسيرها مع بقية الفحوصات والتاريخ المرضي."
            )
        })


    # تنبيه إضافي عند وجود عدة عوامل معًا
    combined_risk_count = 0


    if data["trestbps"] >= 140:

        combined_risk_count += 1


    if data["chol"] >= 200:

        combined_risk_count += 1


    if data["fbs"] is True:

        combined_risk_count += 1


    if data["cp"] in [
        "typical angina",
        "atypical angina"
    ]:

        combined_risk_count += 1


    if data["exang"] is True:

        combined_risk_count += 1


    if combined_risk_count >= 3:

        alerts.append({
            "priority": 1,
            "message": (
                "توجد عدة مؤشرات سريرية وعوامل خطورة متزامنة؛ "
                "يُفضّل إجراء تقييم متكامل بدل تفسير كل متغير بشكل منفصل."
            )
        })


    # ترتيب التنبيهات حسب الأولوية
    alerts.sort(
        key=lambda item: item["priority"]
    )


    # إزالة التكرار
    unique_alerts = []


    for alert in alerts:

        if alert["message"] not in unique_alerts:

            unique_alerts.append(
                alert["message"]
            )


    # الاحتفاظ بأهم أربعة تنبيهات
    return unique_alerts[:4]


# إنشاء توصيات مخصصة للحالة
def generate_recommendations(data, probability, risk_level):

    recommendations = []


    # تحديد التوصية الأساسية حسب مستوى الخطورة
    if risk_level == "مرتفع":

        recommendations.append({
            "priority": 1,
            "category": "التقييم",
            "text": (
                "إعطاء أولوية للتقييم السريري وربط نتيجة النموذج "
                "بالأعراض والتاريخ المرضي والفحص والفحوصات المتاحة."
            )
        })

    elif risk_level == "متوسط":

        recommendations.append({
            "priority": 2,
            "category": "التقييم",
            "text": (
                "إجراء متابعة سريرية لتقييم الأعراض وعوامل الخطورة "
                "وتحديد الحاجة إلى فحوصات إضافية."
            )
        })

    else:

        recommendations.append({
            "priority": 3,
            "category": "المتابعة",
            "text": (
                "الاستمرار في المتابعة الصحية الدورية وعدم الاعتماد "
                "على نتيجة النموذج وحدها لاستبعاد المرض."
            )
        })


    # توصية مخصصة لألم الصدر النموذجي
    if data["cp"] == "typical angina":

        recommendations.append({
            "priority": 1,
            "category": "الأعراض",
            "text": (
                "تقييم ألم الصدر من حيث البداية والمدة والارتباط بالمجهود "
                "والأعراض المصاحبة وتحديد الحاجة إلى تقييم قلبي إضافي."
            )
        })


    # توصية مخصصة لألم الصدر غير النموذجي
    elif data["cp"] == "atypical angina":

        recommendations.append({
            "priority": 2,
            "category": "الأعراض",
            "text": (
                "مراجعة طبيعة ألم الصدر والأعراض المصاحبة "
                "وربطها بالتاريخ المرضي وعوامل الخطورة."
            )
        })


    # توصية عند وجود أعراض أثناء الجهد
    if data["exang"] is True:

        recommendations.append({
            "priority": 1,
            "category": "الجهد",
            "text": (
                "مراجعة الأعراض المرتبطة بالمجهود وربطها بنتائج "
                "اختبار الجهد والتقييم السريري."
            )
        })


    # بناء توصية ضغط الدم
    if data["trestbps"] >= 180:

        recommendations.append({
            "priority": 1,
            "category": "ضغط الدم",
            "text": (
                "إعادة تقييم ضغط الدم ومراجعته سريريًا، "
                "مع متابعة القراءات وتحديد الخطة المناسبة مع الطبيب."
            )
        })

    elif data["trestbps"] >= 140:

        recommendations.append({
            "priority": 2,
            "category": "ضغط الدم",
            "text": (
                "متابعة ضغط الدم بانتظام وتقييم مدى السيطرة عليه "
                "ومناقشة التدخل المناسب مع الطبيب."
            )
        })

    elif data["trestbps"] >= 130:

        recommendations.append({
            "priority": 3,
            "category": "ضغط الدم",
            "text": (
                "متابعة ضغط الدم والمحافظة على نمط حياة يساعد "
                "على التحكم بعوامل الخطورة القلبية."
            )
        })


    # بناء توصية الكوليسترول
    if data["chol"] >= 240:

        recommendations.append({
            "priority": 1,
            "category": "الكوليسترول",
            "text": (
                "مراجعة ملف الدهون وتقييم الكوليسترول ضمن عوامل "
                "الخطر القلبي ومناقشة خطة خفض الخطر مع الطبيب."
            )
        })

    elif data["chol"] >= 200:

        recommendations.append({
            "priority": 2,
            "category": "الكوليسترول",
            "text": (
                "متابعة مستوى الكوليسترول وتحسين العادات الغذائية "
                "ونمط الحياة ومناقشة الخطة المناسبة مع الطبيب."
            )
        })


    # بناء توصية سكر الدم
    if data["fbs"] is True:

        recommendations.append({
            "priority": 2,
            "category": "الاستقلاب",
            "text": (
                "مراجعة حالة سكر الدم وتقييم عوامل الخطورة "
                "الاستقلابية ضمن المتابعة الطبية."
            )
        })


    # بناء توصية للفحوصات القلبية
    cardiac_findings = []


    if data["restecg"] != "normal":

        cardiac_findings.append("تخطيط القلب")


    if data["oldpeak"] > 0:

        cardiac_findings.append("Oldpeak")


    if data["slope"] != "upsloping":

        cardiac_findings.append("ميل ST")


    if data["ca"] > 0:

        cardiac_findings.append("عدد الأوعية")


    if data["thal"] != "normal":

        cardiac_findings.append("Thal")


    # دمج النتائج في توصية واحدة
    if cardiac_findings:

        findings_text = "، ".join(
            cardiac_findings
        )

        recommendations.append({
            "priority": 2,
            "category": "الفحوصات القلبية",
            "text": (
                f"مراجعة نتائج ({findings_text}) وربطها بالفحص السريري "
                "وبقية نتائج الحالة قبل اتخاذ أي قرار."
            )
        })


    # توصية خاصة عند وجود عدة عوامل خطورة
    modifiable_count = 0


    if data["trestbps"] >= 140:

        modifiable_count += 1


    if data["chol"] >= 200:

        modifiable_count += 1


    if data["fbs"] is True:

        modifiable_count += 1


    if modifiable_count >= 2:

        recommendations.append({
            "priority": 2,
            "category": "الوقاية",
            "text": (
                "توجد عدة عوامل خطورة قابلة للتعديل؛ "
                "يُنصح بمراجعة نمط الغذاء والنشاط البدني والوزن "
                "وعوامل الخطر الأخرى ضمن خطة وقائية مناسبة."
            )
        })


    # توصية خاصة بالعمر
    if data["age"] >= 55:

        recommendations.append({
            "priority": 3,
            "category": "الوقاية",
            "text": (
                "نظرًا للعمر وعوامل الخطورة الموجودة، "
                "ينبغي الاستمرار في المتابعة الدورية لعوامل الخطر القلبية."
            )
        })


    # ترتيب التوصيات حسب الأولوية
    recommendations.sort(
        key=lambda item: item["priority"]
    )


    # منع تكرار التوصيات
    unique_recommendations = []

    used_categories = set()


    for recommendation in recommendations:

        category = recommendation["category"]

        text = recommendation["text"]


        if category == "التقييم":

            key = "evaluation"

        elif category == "المتابعة":

            key = "followup"

        else:

            key = category


        if (
            text not in unique_recommendations
            and key not in used_categories
        ):

            unique_recommendations.append(
                text
            )

            used_categories.add(
                key
            )


    # الاحتفاظ بأهم ست توصيات
    unique_recommendations = unique_recommendations[:6]


    # إضافة الضابط الطبي
    unique_recommendations.append(
        "أي قرار دوائي أو علاجي يجب أن يعتمد على التقييم الطبي الكامل وليس على نتيجة النموذج وحدها."
    )


    # إعادة التوصيات
    return unique_recommendations


# الصفحة الرئيسية للنظام
@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# واجهة التنبؤ
@app.route("/predict", methods=["POST"])
def predict():

    # التأكد من إرسال JSON
    if not request.is_json:

        return jsonify({
            "error": "يجب إرسال البيانات بصيغة JSON"
        }), 400


    # قراءة البيانات
    data = request.get_json()


    # التأكد من أن البيانات كائن JSON
    if not isinstance(data, dict):

        return jsonify({
            "error": "بيانات المريض يجب أن تكون على شكل كائن JSON"
        }), 400


    # التحقق من الحقول الناقصة
    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in data
    ]


    if missing_fields:

        return jsonify({
            "error": "توجد حقول مطلوبة غير موجودة",
            "missing_fields": missing_fields
        }), 400


    # التحقق من الحقول غير المسموح بها
    extra_fields = [
        field
        for field in data
        if field not in REQUIRED_FIELDS
    ]


    if extra_fields:

        return jsonify({
            "error": "توجد حقول غير مسموح بها",
            "extra_fields": extra_fields
        }), 400


    # التحقق من القيم الفئوية
    for field, allowed_values in ALLOWED_VALUES.items():

        value = data[field]


        if value not in allowed_values:

            return jsonify({
                "error": f"قيمة {field} غير صحيحة",
                "allowed_values": allowed_values,
                "received_value": value
            }), 400


    # التحقق من القيم الرقمية
    for field, (minimum, maximum) in NUMERIC_RANGES.items():

        value = data[field]


        # التأكد من أن القيمة رقمية
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):

            return jsonify({
                "error": f"قيمة {field} يجب أن تكون رقمية",
                "received_value": value
            }), 400


        # التأكد من أن القيمة داخل النطاق
        if value < minimum or value > maximum:

            return jsonify({
                "error": f"قيمة {field} خارج النطاق المسموح",
                "minimum": minimum,
                "maximum": maximum,
                "received_value": value
            }), 400


    # إنشاء DataFrame بالترتيب نفسه الذي تدرب عليه النموذج
    patient_data = pd.DataFrame(
        [
            [
                data[field]
                for field in REQUIRED_FIELDS
            ]
        ],
        columns=REQUIRED_FIELDS
    )


    # إجراء التنبؤ
    prediction = model.predict(
        patient_data
    )[0]


    # حساب احتمالية الإصابة
    probability = model.predict_proba(
        patient_data
    )[0][1]


    # تحديد مستوى الخطورة
    risk_level = get_risk_level(
        probability
    )


    # إنشاء التشخيص المبدئي
    if prediction == 1:

        diagnosis = (
            "يوجد احتمال للإصابة بمرض القلب"
        )

    else:

        diagnosis = (
            "لا يوجد احتمال مرتفع للإصابة بمرض القلب"
        )


    # استخراج نتائج الحالة
    patient_findings = generate_patient_findings(
        data
    )


    # إنشاء التنبيهات
    doctor_alerts = generate_doctor_alerts(
        data,
        probability,
        risk_level
    )


    # إنشاء التوصيات
    recommendations = generate_recommendations(
        data,
        probability,
        risk_level
    )


    # تجهيز استجابة النظام
    response = {

        "prediction": int(
            prediction
        ),

        "probability": round(
            float(probability),
            4
        ),

        "risk_level": risk_level,

        "diagnosis": diagnosis,

        "patient_findings": patient_findings,

        "doctor_alerts": doctor_alerts,

        "recommendations": recommendations
    }


    # إعادة النتيجة
    return jsonify(
        response
    )


# تشغيل الخادم
if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )