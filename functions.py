import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
# Mesure performance
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

from sklearn.preprocessing import OneHotEncoder

sns.set()


def explore_df(df):
    entries = len(df.index)
    variables = len(df.columns.values.tolist())
    print('Le dataset initial contient {nb_ind} individus et {nb_var} variables.'.format(nb_ind=entries,
                                                                                         nb_var=variables))
    dtype_count = df.dtypes.value_counts()
    types_diff = len(dtype_count)

    for i in range(types_diff):
        if i == 0:
            print('Les variables sont de {types_diff} types :'.format(types_diff=types_diff))

        print('- {nb_type} variables de type {d_type}'.format(nb_type=dtype_count.iloc[i], d_type=dtype_count.index[i]))


def kernel_plots(df, x_size=3):
    df_num = df.select_dtypes(include='number')
    plots_nb = df_num.shape[1]

    if plots_nb != 0:
        plots_titles = df_num.columns

        fig, axs = plt.subplots(int(plots_nb / x_size) + 1, x_size, figsize=(x_size * 4, (plots_nb / x_size) * 4))
        fig.suptitle('Analyse des variables numériques')

        for i in range(plots_nb):
            sns.kdeplot(df_num.iloc[:, i], ax=axs[int(i / x_size)][i - int(i / x_size) * x_size])

        for i in range(((int(plots_nb / x_size) + 1) * x_size) - plots_nb):
            axs[-1][-1 - i].axes.set_visible(False)

        plt.show()

    else:
        print('Pas de données numériques')


def heatmap_plot(df):
    df_num = df.select_dtypes(include='number')
    sns.heatmap(df_num.corr(), annot=True, fmt=".2f")
    plt.xticks(rotation=35)
    plt.title('Matrice des corrélations')


def isna_nunique(df):
    # Quel est le % de valeurs manquantes par colonne ?
    # Combien y a-t-il de valeurs différentes par colonne ?

    return pd.concat([pd.Series(round(df.isna().mean()*100, 0), name='% valeurs manquantes'),
                      pd.Series(round(df.isna().sum(), 0), name='# valeurs manquantes'),
                      pd.Series(df.nunique(), name='# valeurs différentes')], axis=1)


def etat_df(df, message, entries=None, variables=None):
    entries_0 = entries
    variables_0 = variables

    entries = len(df.index)
    variables = len(df.columns.values.tolist())
    
    print(message + ', le dataset contient {nb_ind} ({nb_ind_0} initialement) individus et {nb_var} ({nb_var_0} initialement) variables.'.format(nb_ind_0=entries_0, nb_var_0=variables_0, nb_ind=entries, nb_var=variables))

    return entries, variables


def custom_search(X, y, model_list, modeles, scalers, features_scale, parametres_distributions, scoring, niter, prnt, random_state=42, verbose=4):
    # Création des ensembles de train et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)

    cv_result = pd.DataFrame(
        columns=['Regressor', 'Scaler', 'Best Score', 'Best Estimator', 'Best Params', 'Mean Fit Time', 'coefficient_of_dermination',
                 'rmse', 'mape'])
    y_pred_test = pd.DataFrame(index=y_test.index)

    i = 0
    total = len(scalers.keys()) * len(model_list)

    for modele in modeles.keys():
        if modele in model_list:
            for scaler in scalers.keys():

                i = i + 1
                if prnt: print(f'{i} / {total}', modele, scaler)

                col_transformer = ColumnTransformer(
                    [
                        (scaler, scalers[scaler], features_scale)
                    ],
                    remainder='passthrough')

                pipeline = Pipeline(steps=[('transformer', col_transformer), (modele, modeles[modele])])

                if modele == 'linreg':
                    rnd_srch = RandomizedSearchCV(pipeline, param_distributions=parametres_distributions[modele], cv=5,
                                                  scoring=scoring, refit='r2', n_iter=1, return_train_score=True
                                                  , verbose=verbose
                                                  )
                else:
                    rnd_srch = RandomizedSearchCV(pipeline, param_distributions=parametres_distributions[modele], cv=5,
                                                  scoring=scoring, refit='r2', n_iter=niter, return_train_score=True,
                                                  verbose=verbose)

                # Revoir l'utilisation des scores
                # CV => choisir un score sur lequel on veut optimiser (R2)
                # On s'intéresse au score de la validation pour choisir le modèle final
                # On teste le best estimator sur le Test pour regarder le under/over fitting
                # Afficher aussi les autres scores

                rnd_srch.fit(X_train, y_train)
                if prnt: print('cv_results_', pd.DataFrame(rnd_srch.cv_results_))

                best_estimator = rnd_srch.best_estimator_
                if prnt: print('best_estimator', best_estimator)

                best_score = round(rnd_srch.best_score_, 4)  # Mean cross-validated score of the best_estimator.
                if prnt: print(best_score)

                best_params = rnd_srch.best_params_

                mean_fit_time = rnd_srch.cv_results_['mean_fit_time'].mean()

                y_pred_test.loc[:, scaler + '_' + modele] = rnd_srch.predict(X_test)

                cv_result.loc[scaler + '_' + modele] = [modele, scaler, best_score, best_estimator,
                                                        best_params, mean_fit_time,
                                                        r2_score(y_test, y_pred_test.loc[:, scaler + '_' + modele]),
                                                        mean_squared_error(y_test,
                                                                           y_pred_test.loc[:, scaler + '_' + modele]),
                                                        mean_absolute_percentage_error(y_test, y_pred_test.loc[:,
                                                                                               scaler + '_' + modele])]

    return cv_result


def plot_result(result, values_list):
    models = result.index.values
    models_values = {
        key: value for key, value in zip(
            values_list,
            [result[values_list[i]].values for i in range(len(values_list))]
        )
    }

    x = np.arange(len(models))  # the label locations
    width = 0.35  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(15, 10))

    for attribute, measurement in models_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Values')
    ax.set_title('Scores by model')
    ax.set_xticks(x + width, models)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.show()


def year_transformer(df, target_name, new_name):
    df[new_name] = 2024 - df[target_name]
    df = df.drop([target_name], axis=1)
    return df


def feat_encode(df, target, drop_name):
    property_types = df[target].unique().tolist()

    property_types_other = property_types.copy()
    property_types_other.remove(drop_name)

    ohe = OneHotEncoder(categories=[property_types], dtype=int, drop=[drop_name], sparse_output=False)

    df_encode = pd.DataFrame(ohe.fit_transform(df[target].values.reshape(-1, 1)), columns=property_types_other, index=df.index)

    features_remainder = df.columns.tolist()
    features_remainder.remove(target)

    return pd.concat([df_encode, df.loc[:, features_remainder]], axis=1)


def custom_grid_search(X, y, model_list, modeles, scalers, features_scale, parametres, scoring, prnt, random_state=42, verbose=4):
    # Création des ensembles de train et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)

    cv_result = pd.DataFrame(
        columns=['Regressor', 'Scaler', 'Best Score', 'Best Estimator', 'Best Params', 'Mean Fit Time', 'coefficient_of_dermination',
                 'rmse', 'mape'])
    y_pred_test = pd.DataFrame(index=y_test.index)

    i = 0
    total = len(scalers.keys()) * len(model_list)

    for modele in modeles.keys():
        if modele in model_list:
            for scaler in scalers.keys():

                i = i + 1
                if prnt: print(f'{i} / {total}', modele, scaler)

                col_transformer = ColumnTransformer(
                    [
                        (scaler, scalers[scaler], features_scale)
                    ],
                    remainder='passthrough')

                pipeline = Pipeline(steps=[('transformer', col_transformer), (modele, modeles[modele])])

                grd_srch = GridSearchCV(pipeline, param_grid=parametres[modele], cv=5,
                                        scoring=scoring, refit='r2', return_train_score=True, verbose=verbose
                                        )

                # Revoir l'utilisation des scores
                # CV => choisir un score sur lequel on veut optimiser (R2)
                # On s'intéresse au score de la validation pour choisir le modèle final
                # On teste le best estimator sur le Test pour regarder le under/over fitting
                # Afficher aussi les autres scores

                grd_srch.fit(X_train, y_train)
                if prnt: print('cv_results_', pd.DataFrame(grd_srch.cv_results_))

                best_estimator = grd_srch.best_estimator_
                if prnt: print('best_estimator', best_estimator)

                best_score = round(grd_srch.best_score_, 4)  # Mean cross-validated score of the best_estimator.
                if prnt: print(best_score)

                best_params = grd_srch.best_params_

                mean_fit_time = grd_srch.cv_results_['mean_fit_time'].mean()

                y_pred_test.loc[:, scaler + '_' + modele] = grd_srch.predict(X_test)

                cv_result.loc[scaler + '_' + modele] = [modele, scaler, best_score, best_estimator,
                                                        best_params, mean_fit_time,
                                                        r2_score(y_test, y_pred_test.loc[:, scaler + '_' + modele]),
                                                        mean_squared_error(y_test,
                                                                           y_pred_test.loc[:, scaler + '_' + modele]),
                                                        mean_absolute_percentage_error(y_test, y_pred_test.loc[:,
                                                                                               scaler + '_' + modele])]

    return cv_result
